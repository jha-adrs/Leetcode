#include<iostream>
#include <bits/stdc++.h>
using namespace std;
/*
When u delete a index the next one's index is decreased
If equal */
class Solution{
    public:
    int removeDuplicates(vector<int>&nums)
    {
        int count =1; // We take 1 so that the first one is already included
        // Also we are not substituing 0th index
        // Check i and i-1 th element
        for (int i = 1; i < nums.size(); i++)
        {
            if(nums[i] != nums[count-1]){
                nums[count] = nums[i];
                count++;
            }
            // If it is equal then it gets replaced in the next iteration
        }
        

        return count;


    }
};

 