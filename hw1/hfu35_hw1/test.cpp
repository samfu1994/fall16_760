int func(vector<int> & nums){
	stack<int> s;
	int l = nums.size();
	int minval = l, maxval = -1;
	for(int i = 0; i < l; i++){
		while(!s.empty() || nums[s.top()] > nums[i]){
			minval = min(minval, nums[s.top()]);
			s.pop();
		}
		if(i != 0 && s.empty()) break;
		s.push(i);
	}
	if(minval == l){
		return 0;
	}
	while(!s.empty()) s.pop();
	for(int i = l - 1; i >= 0; i--){
		while(!s.empty() || nums[s.top()]< nums[i]){
			maxval = max(maxval, nums[s.top()]);
			s.pop();
		}
		if(i != l - 1 && s.empty()) break;
		s.push(i);
	}
	return maxval - minval + 1;
}