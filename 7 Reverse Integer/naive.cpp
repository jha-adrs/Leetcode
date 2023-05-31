// Goal is to input a signed 32-bit integer and return the reverse of that integer. 
//If the reversed integer overflows, return 0.
// Apparetly u dont need to exclude the -ve bit from the reversing number as it works on both +ve and -ve numbers
// 123 % 10 = 3
// -123 % 10 = -3 So same thing
#include<iostream>
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    int reverse(int x) {
        int rev= 0;
        int state=1;
        if(x<0){
            state = -1;
        }
        //x = abs(x);
        while(x != 0){
            int d = x%10;
            
            if(rev > INT_MAX/10 || (rev == INT_MAX/10 && d >7)) return 0; // overflow in next 
            if( rev < INT_MIN/10 || (rev == INT_MIN/10 && d <  -8)) return 0;// underflow-
            rev = rev*10+d;
            x/=10;
        }
        return rev;

    }

};
int main()
{
    Solution ss;
    signed int x;
    cin>>x;
    cout<<ss.reverse(x);

    return 0;
}