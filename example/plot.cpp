

int main(){
    [[benchmark("foo")]]{
        [[using NAME:  list("foo", "bar")]];
        [[using COUNT: range(10)]];
        [[use(NAME, COUNT)]];
        
        [[plot]]{
            [[x_axis(COUNT)]];
            [[y_axis($.elapsed_ms)]];
        }
    }
}