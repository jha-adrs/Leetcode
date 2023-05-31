// Lol u cant use extra space :)
#include<iostream>
#include <bits/stdc++.h>
using namespace std;

class Solution{
    public:
    int removeDuplicates(vector<int>&nums)
    {
        unordered_map<int,int> mp;
        int count =0;
        for(int i =0; i<nums.size(); i++)
        {
            if(mp[nums[i]]){
                count++;
                nums.erase(nums.begin()+i);
            }
            else{
                mp[nums[i]] = 1;
            }
        }
        return count;


    }
};

int main()
{
    
    return 0;
}