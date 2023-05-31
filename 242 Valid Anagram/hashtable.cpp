#include<iostream>
#include <bits/stdc++.h>
using namespace std;
// Best in runtime O(n)
class Solution
{
    public:
    bool isAnagram(string s, string t)
    {
        unordered_map<char,int> table;
        if(s.length()!= t.length())
        return false;
        if(s==t)
        {
            return true;
        }
        for (int i = 0; i < s.length(); i++)
        {
            table[s[i]]++;
        }
        for (int i=0; i< t.length(); i++)
        {
            
            table[t[i]]--;
            if(table[t[i]] <0) 
            {
                return false;
            }
        }
        
        return true;
    }
};
int main()
{
    
    return 0;
}

/*
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
*/