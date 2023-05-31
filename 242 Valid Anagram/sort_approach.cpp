#include<iostream>
#include <bits/stdc++.h>
using namespace std;
// Slower in runtime, better in memory O(nlogn) 
int main()
{
    
    return 0;
}
class Solution
{

public:
public:
    bool isAnagram(string s, string t)
    {
        if(t.length() != s.length())
        {
            return false;

        }
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());
        return s==t;
    }
};

