from . import report
from multiprocessing import JoinableQueue, Manager, Pipe, Process, Queue, current_process, cpu_count
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
import queue
import signal
from tempfile import TemporaryDirectory
import time
import traceback
from typing import Any, Iterable, Optional

import click
from colorama import Fore, Style
import psutil

from .parser import Test, TranslationUnit
from .compilers import Compiler
from .util import ExecResult, decorated, fnv1a, run, to_base58


def progressbar_status(shutdown_event: EventClass, output_queue: Queue, total: int):
    with click.progressbar(length=total, show_pos=True, show_percent=False, show_eta=True) as progressbar:
        while not shutdown_event.is_set():
            # use update to enable eta calculation
            progressbar.update(output_queue.qsize() - progressbar.pos)
            time.sleep(0.1)

        # finally set progressbar to 100%
        progressbar.update(total - progressbar.pos)


class PendingRun:
    def __init__(self, configuration: Compiler, source: Path, test_name: str, variables: dict, outpath: Path):
        self.compiler = configuration
        self.source = source
        self.test_name = test_name
        self.variables = variables
        self.outpath = outpath
        self.command = configuration.compile_command(source, outpath, test_name, variables=variables)
        self.result: Optional[ExecResult] = None

    def __str__(self):
        return ' '.join(self.command)

    __repr__ = __str__

    def run(self):
        self.outpath.mkdir(exist_ok=True, parents=True)
        for _ in range(5):
            result = run(self.command)
            if not self.result or self.result.elapsed_ms > result.elapsed_ms:
                # pick the best result out of all reruns
                self.result = result
        return self

    def run_assertions(self):
        assert self.result is not None, "Must run test before running assertions against it"
        return self.compiler.run_assertions()


class Cancelled(Exception):
    def __init__(self, reason: str, run: Optional[PendingRun] = None):
        self.reason = reason
        self.run = run

    def explain(self):
        def stringify():
            yield decorated(f"Cancelled: {self.reason}", fg=Fore.RED)
            if self.run is None:
                return
            yield f"  Command: {' '.join(self.run.command)}"

            if self.run.result is None:
                return
            yield f"  Return Code: {decorated(self.run.result.returncode, fg=Fore.RED)}" \
                f" (expected {decorated(self.run.compiler.expected_return_code, fg=Fore.GREEN)})"
            yield decorated("STDERR", style=Style.BRIGHT)
            yield self.run.result.stderr
            yield decorated("STDOUT", style=Style.BRIGHT)
            yield self.run.result.stdout
        return '\n'.join(stringify())

    def __str__(self):
        return self.reason


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
        # TODO pin actual subprocesses to cpu

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
        self.num_jobs = num_jobs or cpu_count()
        self.manager = Manager()
        self.tasks = self.manager.Queue()
        self.shutdown_event = self.manager.Event()
        self.process_event = self.manager.Event()

        if pin_cpu is None:
            pin_cpu = [None for _ in range(self.num_jobs)]
            self.pin_cpu = pin_cpu

        assert len(pin_cpu) == self.num_jobs, "Must specify core for every job to be pinned to"

        self.workers = [Worker(target=worker_task,
                               args=(self.shutdown_event, self.process_event, self.tasks, pin_cpu[idx]))
                        for idx in range(self.num_jobs)]

    def start(self):
        for worker in self.workers:
            worker.start()

    def join(self):
        self.shutdown_event.set()
        for worker in self.workers:
            worker.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type_, value, traceback):
        self.join()

    def clear_tasks(self):
        while not self.tasks.empty():
            try:
                self.tasks.get(block=False)
            except queue.Empty:
                break
        assert self.tasks.empty()

    def run_tasks(self, tasks: Iterable[Any], timeout: Optional[float] = None, cancel_on_error=True):
        if not isinstance(tasks, list):
            # evaluate task list - its size must be known
            tasks = list(tasks)
        assert self.tasks.empty()

        def cancel(signum, frame):
            self.shutdown_event.set()
            for worker in self.workers:
                worker.kill()

            raise SystemExit(1)

        sigint_handler = signal.signal(signal.SIGINT, cancel)
        try:
            results = self.manager.Queue()

            # enqueue tasks
            for task in tasks:
                self.tasks.put((results, task), timeout=timeout)

            self.process_event.set()

            progressbar = click.progressbar(length=len(tasks), show_pos=True, show_percent=False, show_eta=True)
            collected_results = []
            old_len = 0
            while len(collected_results) != len(tasks):
                try:
                    evaluated: PendingRun = results.get(timeout=timeout)
                    collected_results.append(evaluated)

                    assert evaluated.result is not None
                    if cancel_on_error and evaluated.result.returncode != evaluated.compiler.expected_return_code:
                        print("error: task failed - cancelling")
                        self.process_event.clear()
                        self.clear_tasks()
                        raise Cancelled("Task failed.", evaluated)

                except queue.Empty:
                    # cannot dequeue any more from results

                    # TODO
                    dead_workers = []
                    for idx, worker in enumerate(self.workers):
                        if not worker.has_exception:
                            continue
                        dead_workers.append((idx, worker.exception))

                    if dead_workers:
                        # TODO
                        print(dead_workers)
                        raise Cancelled("All workers died")

                if (length := len(collected_results)) != old_len:
                    progressbar.update(length - old_len)
                    old_len = length
                else:
                    # no new data received - sleep and try again
                    time.sleep(0.1)

            print()  # print a newline after the progressbar
            return collected_results

        finally:
            self.process_event.clear()
            signal.signal(signal.SIGINT, sigint_handler)


class Runner:
    def __init__(self, pool: Optional[Pool] = None, keep_files: bool = False):
        self.pool = pool
        self.build_dir = TemporaryDirectory("crashbench", delete=not keep_files)

    def compile_commands(self, source: Path, test: Test, compiler: Compiler):
        assert source.exists()
        outpath = Path(self.build_dir.name) / test.name / to_base58(compiler.hash)
        for variables in test.runs:
            yield PendingRun(compiler, source, test.name, variables, outpath)

    def run_commands(self, compile_commands: Iterable[PendingRun]):
        if self.pool is None:
            for command in compile_commands:
                yield command.run()
        else:
            yield from self.pool.run_tasks(compile_commands)

    def run_test(self, source: Path, test: Test):
        for configuration in test.settings.effective_configurations():
            print(configuration)
            compile_commands = list(self.compile_commands(source, test, configuration))
            yield from self.run_commands(compile_commands)

    def run(self, source_path: Path, dry: bool = False):
        tu = TranslationUnit.from_file(source_path)
        processed_path = Path(self.build_dir.name) / source_path.name
        processed_path.write_text(tu.source)
        output = report.Report(source_path)

        try:
            for test in tu.tests:
                print(f"Running test {decorated(test.name, style=Style.BRIGHT)}")
                results = []
                for configuration in test.settings.effective_configurations():
                    print(configuration)
                    compile_commands = list(self.compile_commands(processed_path, test, configuration))

                    if dry:
                        for command in compile_commands:
                            print(command)
                        continue

                    for processed in self.run_commands(compile_commands):
                        var_hash = to_base58(fnv1a(processed.variables))
                        extra_files = list(processed.compiler.expand_extra_files(processed.outpath, var_hash))
                        result = report.Result(report.CompilerInfo(processed.compiler), processed.variables, processed.result, extra_files)
                        results.append(result)
                        output.add_result(test.name, result)

                        # print(f"{run.compiler} {run.variables!s: <50}: {int(run.result.elapsed_ms)}ms")
                        # run.run_assertions()

                for output_action in test.output_actions:
                    output_action(results)
        except Cancelled as cancellation:
            print(cancellation.explain())
            return

        return output