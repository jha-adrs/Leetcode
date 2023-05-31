/*
Not the best one
Explanation: It uses an unordered hasmap (ordered hashmap may )
First it iterates through all the elements in 1st vector and increases their freq in unordered hash
Second it iterates through the second set of vector array, 
if it already exists decrease it and add it to the final list */
#include<iostream>
#include <bits/stdc++.h>
using namespace std;
class Solution
{
    public:
    vector<int> intersect(vector<int>&nums1, vector<int> &nums2)
    {
        vector<int> res;
        unordered_map<int,int> freq_table; // creates a hashmap linear

        for (int i = 0; i <nums1.size(); i++)
        {
            freq_table[nums1[i]]++;
        }
        for (int i = 0; i < nums2.size(); i++)
        {
            if(freq_table[nums2[i]] >0)
            {
                res.push_back(nums2[i]);
                freq_table[nums2[i]]--;
            }
            
        }
        return res;
        
        
    }
}; 

int main()
{
    vector<int> init1 = {3,1,2};
    vector<int> init2 = {1,1};
    Solution sol;
    vector<int> res = sol.intersect(init1, init2);
    cout << "[";
    for (int i = 0; i < res.size(); i++)
    {
        cout << res[i] << ", ";
    }
    cout << "]";
    return 0;
}