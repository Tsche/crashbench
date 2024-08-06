int main(){
  [[test("foo")]] {
    [[standard(">=20")]];
    [[using X: range(10)]];
    int x = X;
  }
}