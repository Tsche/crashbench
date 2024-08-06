from multiprocessing import JoinableQueue, Manager, Pipe, Process, Queue, Event, current_process
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from pprint import pprint
import queue
import signal
from tempfile import TemporaryDirectory
import time
import traceback
from typing import Any, Optional

import click
import psutil

from .parser import Test, TranslationUnit
from .compilers import Compiler
from .util import Result, fnv1a, run, to_base58

def progressbar_status(shutdown_event: EventClass, output_queue: Queue, total: int):
    with click.progressbar(length=total, show_pos=True, show_percent=False, show_eta=True) as progressbar:
        while not shutdown_event.is_set():
            # use update to enable eta calculation
            progressbar.update(output_queue.qsize() - progressbar.pos)
            time.sleep(0.1)

        # finally set progressbar to 100%
        progressbar.update(total - progressbar.pos)


class Worker(Process):
    def __init__(self, target, args) -> None:
        super().__init__(target=target, args=[*args], daemon=True)
        self._parent, self._child = Pipe()
        self._exception = None

    def run(self):
        try:
            super().run()
            self._child.send(None)
        except KeyboardInterrupt:
            self.terminate()
        except Exception as exc:
            trace = traceback.format_exc()
            print(f"error in {current_process().name}: {exc}")
            self._child.send((exc, trace))


    @property
    def exception(self):
        if self._parent.poll():
            self._exception = self._parent.recv()
        return self._exception

    @property
    def has_exception(self):
        return (self._exception or self.exception) is not None

def worker_task(shutdown_event: EventClass,
                process_event: EventClass,
                input_queue: JoinableQueue,
                pin: Optional[int]):
    # pin to cpu if possible
    if pin is not None:
        this_process = psutil.Process()
        this_process.cpu_affinity([pin])

    while not shutdown_event.is_set():
        if not process_event.wait(timeout=0.1):
            # spin until instructed to process the queue
            continue

        try:
            output_queue, task = input_queue.get(block=False)
            value = task.run()
            output_queue.put(value)
            input_queue.task_done()
        except queue.Empty:
            pass


class Pool:
    def __init__(self, num_jobs: int, pin_cpu: Optional[list[int | None]] = None):
        self.num_jobs = num_jobs
        self.manager = Manager()
        self.tasks: JoinableQueue = JoinableQueue()
        # self.results: Queue = Queue()
        self.shutdown_event = self.manager.Event()
        self.process_event = self.manager.Event()

        if pin_cpu is None:
            pin_cpu = [None for _ in range(num_jobs)]
            self.pin_cpu = pin_cpu

        assert len(pin_cpu) == num_jobs, "Must specify core for every job to be pinned to"

        self.workers = [Worker(target=worker_task,
                               args=(self.shutdown_event, self.process_event, self.tasks, pin_cpu[idx]))
                        for idx in range(self.num_jobs)]

    def start(self):
        for worker in self.workers:
            worker.start()

    def join(self):
        self.tasks.cancel_join_thread()
        self.shutdown_event.set()
        for worker in self.workers:
            worker.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type_, value, traceback):
        self.join()
        self.tasks.cancel_join_thread()
        self.shutdown_event.set()
        for worker in self.workers:
            worker.kill()

    def run_tasks(self, tasks: list[Any], timeout: Optional[float] = None, observer=None):
        # assert self.results.empty()
        assert self.tasks.empty()
        shutdown_observer = self.manager.Event()

        def cancel(signum, frame):
            shutdown_observer.set()
            self.shutdown_event.set()
            self.tasks.cancel_join_thread()
            for worker in self.workers:
                worker.kill()

            raise SystemExit(1)

        sigint_handler = signal.signal(signal.SIGINT, cancel)
        try:
            results = self.manager.Queue()
            if observer is not None:
                observer_proc = Worker(target=observer, args=(shutdown_observer, results, len(tasks)))
                observer_proc.start()

            # enqueue tasks
            for task in tasks:
                self.tasks.put((results, task), timeout=timeout)

            while not self.tasks.empty():
                # TODO restart dead workers?
                if all(worker.has_exception for worker in self.workers):
                    # all workers died, cancel
                    raise RuntimeError([worker.exception for worker in self.workers])
                time.sleep(0.1)

            # wait for all tasks to complete
            self.tasks.join()
            shutdown_observer.set()
            observer_proc.join()

            # retrieve results
            collected_results = []
            min_expected_amount = len(tasks) - [worker.has_exception for worker in self.workers].count(True)
            while len(collected_results) != len(tasks):
                try:
                    evaluated: PendingRun = results.get(timeout=timeout)
                    collected_results.append(evaluated)
                except queue.Empty:
                    # cannot dequeue any more from results
                    if len(collected_results) >= min_expected_amount:
                        print("warning: did not get full amount of results, some workers errored")
                        return collected_results

            assert results.empty()
            return collected_results

        finally:
            signal.signal(signal.SIGINT, sigint_handler)

class PendingRun:
    def __init__(self, configuration: Compiler, source: Path, test_name: str, variables: dict, outpath: Path):
        self.compiler = configuration
        self.source = source
        self.test_name = test_name
        self.variables = variables
        self.outpath = outpath
        self.command = configuration.compile_command(source, outpath, test_name, variables=variables)
        self.result: Optional[Result] = None

    def __str__(self):
        return ' '.join(self.command)

    __repr__ = __str__

    def run(self):
        self.result = run(self.command)
        return self

    def run_assertions(self):
        assert self.result is not None, "Must run test before running assertions against it"
        return self.compiler.run_assertions()

class Runner:
    def __init__(self, pool: Optional[Pool] = None, keep_files: bool = False):
        self.pool = pool
        self.build_dir = TemporaryDirectory("crashbench", delete=not keep_files)

        # if platform.system() != 'Linux' and pin_cpu is not None:
        #     logging.warning("CPU pinning is currently only supported for Linux. This setting will be ignored.")
        #     self.pin_cpu = None

    def compile_commands(self, source: Path, test: Test):
        assert source.exists()
        for configuration in test.settings.effective_configurations():
            config_hash = fnv1a((configuration.path, configuration.get_compiler_info(configuration.path), configuration.options))
            outpath = Path(self.build_dir.name) / test.name / to_base58(config_hash)

            for variables in test.runs:
                yield PendingRun(configuration, source, test.name, variables, outpath)

    def run_test(self, source: Path, test: Test):
        compile_commands = list(self.compile_commands(source, test))
        if self.pool is None:
            for command in compile_commands:
                yield command.run()
        else:
            yield from self.pool.run_tasks(compile_commands, observer=progressbar_status)


    def run(self, source_path: Path, dry: bool = False):
        tu = TranslationUnit.from_file(source_path)
        processed_path = Path(self.build_dir.name) / source_path.name
        processed_path.write_text(tu.source)

        for test in tu.tests:
            if dry:
                runs = list(self.compile_commands(processed_path, test))
                for run in runs:
                    print(run)
                continue

            results = list(self.run_test(processed_path, test))
            for result in results:
                print("RESULT")
                print(result.result)
                result.run_assertions()
