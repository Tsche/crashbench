
int main(){
    [[benchmark("foo")]]{
        [[using COUNT: range(10)]];
        [[use(COUNT)]];
        
        [[output]]{
            [[render("234")]];
        }
    }
}