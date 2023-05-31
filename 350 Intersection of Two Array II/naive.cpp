// Doesnt work
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    vector<int> intersect(vector<int> &nums1, vector<int> &nums2)
    {
        vector<int> res;
        int min_val;
        int c=0;
        vector<int> *p = new vector<int>();
        vector<int> *q;
        if (nums1.size() > nums2.size())
        {
            min_val = nums2.size();
            p = &nums2;
            q = &nums1;
        }
        else
        {
            min_val = nums1.size();
            p = &nums1;
            q = &nums2;
        }

        for (int i = 0; i < min_val; i++)
        {
            for (int j = 0; j < q->size(); j++)
            {
                if ((*q)[j] == (*p)[i])
                {
                    res.push_back((*q)[j]);
                    cout<<res[c];
                    c++;
                    break;
                }
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

/*Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.



Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Explanation: [9,4] is also accepted.


Constraints:

1 <= nums1.length, nums2.length <= 1000
0 <= nums1[i], nums2[i] <= 1000


Follow up:

What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?*/