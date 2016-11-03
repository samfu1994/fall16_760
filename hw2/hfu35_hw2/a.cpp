    void func(vector<vector<int> > & res, vector<int> vec, int n, int start){
        int sq = sqrt(n);
        vector<int> v;
        for(int i = start; i <= sq; i++){
            if(n % i) continue;
            v = vec;
            v.push_back(i);
            func(res, v, n / i, i);
            v.push_back(n / i);
            res.push_back(v);
        }
    }
    vector<vector<int>> getFactors(int n) {
        vector<vector<int> > res;
        vector<int> vec;
        func(res, vec, n, 2);
        return res;
    }
