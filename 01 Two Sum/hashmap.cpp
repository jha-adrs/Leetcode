// Using unordered map in stl lib which uses hashmap
/*
Runtime: 8ms
Beats: 98%
*/
#include<iostream>
#include <bits/stdc++.h>
using namespace std;
class Solution
{

public:
    vector<int> twoSum(vector<int> nums, int target)
    {
        unordered_map<int, int> nn;
        int remSum;
        int  l = nums.size();
        for(int i = 0 ; i<l; i++)
        {
            remSum = target- nums[i];
            
            if(nn.find(remSum) != nn.end())
            {
                
                return {i,nn[remSum]};
            }
            nn[nums[i]] = i;
        }
        return {-1,-1};
    }
};


    int main()
{
    Solution ss;
    vector<int> num = {1,2,3,4,5};
    int target = 5;
    vector<int> result = ss.twoSum(num,target);
    for(int x : result)
    {
        cout<<x<<" ";
    }
    return  0;
}

