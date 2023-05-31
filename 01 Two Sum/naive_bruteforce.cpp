#include <vector>
#include<iostream>
using namespace std;

class Solution
{

public:
    vector<int> twoSum(vector<int> nums, int target)
    {
        int remsum;
        vector <int> res={-1,-1};
        for( int i = 0 ; i<nums.size(); i++)
        {
            remsum = target - nums[i];
            for(int j = 0 ; j<nums.size(); j++)
            {
                if(nums[j]==remsum)
                {
                    res ={i,j};
                    return res;
                }
            }
        }
        return res;
    }       
    
};

int main()
{
    Solution ss;
    vector<int> num = {1,2,3,4,5};
    int target = 4;
    vector<int> result = ss.twoSum(num,target);
    for(int x : result)
    {
        cout<<x;
    }
}