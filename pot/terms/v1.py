def v1(idx,d1,d2,d3):
	return d1[*idx,0] * d1[*idx,0] + d1[*idx,1] * d1[*idx,1] + d1[*idx,2] * d1[*idx,2] + d1[*idx,3] * d1[*idx,3] + d2[*idx,0] * d2[*idx,0] + d2[*idx,1] * d2[*idx,1] + d2[*idx,2] * d2[*idx,2] + d2[*idx,3] * d2[*idx,3] + d3[*idx,0] * d3[*idx,0] + d3[*idx,1] * d3[*idx,1] + d3[*idx,2] * d3[*idx,2] + d3[*idx,3] * d3[*idx,3]
