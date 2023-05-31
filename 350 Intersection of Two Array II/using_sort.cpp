// The hashmap is faster but takes more memory
// Using this takes less memory but more time
#include<iostream>
#include <bits/stdc++.h>
using namespace std;
class Solution
{
    public:
    vector<int> intersect(vector<int>&nums1, vector<int> &nums2)
    {
        vector<int> res;
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        int i = 0;
        int j = 0;
        while(i<nums1.size() && j<nums2.size())
        {
            if(nums1[i] == nums2[j])
            {
                res.push_back(nums1[i]);
                i++;j++;
            }
            else if(nums1[i] > nums2[j])
            {
                j++;
            }
            else if(nums1[i] < nums2[j])
            {
                i++;
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