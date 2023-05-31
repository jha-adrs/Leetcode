#include<iostream>
#include <bits/stdc++.h>
using namespace std;
 

class Solution {
public:
    bool containsDuplicate(vector<int>& nums) 
    {
        int l = nums.size();
        sort(nums.begin(), nums.end());
        for(int  i = 0 ; i<l-1; i++)
        {
            if(nums[i] == nums[i+1])
            {
                return true;
            }
        }
        return false;
    }
};
int main()
{
    
    return 0;
}