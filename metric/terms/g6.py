def g6(idx,DA,DB,d1,d2,d3):
	return (DA[*idx,0] * DB[*idx,0] + DA[*idx,1] * DB[*idx,1] + DA[*idx,2] * DB[*idx,2] + DA[*idx,3] * DB[*idx,3]) * (d1[*idx,0] * d1[*idx,0] * d1[*idx,0] * d1[*idx,0] + d1[*idx,0] * d1[*idx,0] * d1[*idx,1] * d1[*idx,1] + d1[*idx,0] * d1[*idx,0] * d1[*idx,2] * d1[*idx,2] + d1[*idx,0] * d1[*idx,0] * d1[*idx,3] * d1[*idx,3] + d1[*idx,0] * d2[*idx,0] * d1[*idx,0] * d2[*idx,0] + d1[*idx,0] * d2[*idx,0] * d1[*idx,1] * d2[*idx,1] + d1[*idx,0] * d2[*idx,0] * d1[*idx,2] * d2[*idx,2] + d1[*idx,0] * d2[*idx,0] * d1[*idx,3] * d2[*idx,3] + d1[*idx,0] * d3[*idx,0] * d1[*idx,0] * d3[*idx,0] + d1[*idx,0] * d3[*idx,0] * d1[*idx,1] * d3[*idx,1] + d1[*idx,0] * d3[*idx,0] * d1[*idx,2] * d3[*idx,2] + d1[*idx,0] * d3[*idx,0] * d1[*idx,3] * d3[*idx,3] + d1[*idx,1] * d1[*idx,1] * d1[*idx,0] * d1[*idx,0] + d1[*idx,1] * d1[*idx,1] * d1[*idx,1] * d1[*idx,1] + d1[*idx,1] * d1[*idx,1] * d1[*idx,2] * d1[*idx,2] + d1[*idx,1] * d1[*idx,1] * d1[*idx,3] * d1[*idx,3] + d1[*idx,1] * d2[*idx,1] * d1[*idx,0] * d2[*idx,0] + d1[*idx,1] * d2[*idx,1] * d1[*idx,1] * d2[*idx,1] + d1[*idx,1] * d2[*idx,1] * d1[*idx,2] * d2[*idx,2] + d1[*idx,1] * d2[*idx,1] * d1[*idx,3] * d2[*idx,3] + d1[*idx,1] * d3[*idx,1] * d1[*idx,0] * d3[*idx,0] + d1[*idx,1] * d3[*idx,1] * d1[*idx,1] * d3[*idx,1] + d1[*idx,1] * d3[*idx,1] * d1[*idx,2] * d3[*idx,2] + d1[*idx,1] * d3[*idx,1] * d1[*idx,3] * d3[*idx,3] + d1[*idx,2] * d1[*idx,2] * d1[*idx,0] * d1[*idx,0] + d1[*idx,2] * d1[*idx,2] * d1[*idx,1] * d1[*idx,1] + d1[*idx,2] * d1[*idx,2] * d1[*idx,2] * d1[*idx,2] + d1[*idx,2] * d1[*idx,2] * d1[*idx,3] * d1[*idx,3] + d1[*idx,2] * d2[*idx,2] * d1[*idx,0] * d2[*idx,0] + d1[*idx,2] * d2[*idx,2] * d1[*idx,1] * d2[*idx,1] + d1[*idx,2] * d2[*idx,2] * d1[*idx,2] * d2[*idx,2] + d1[*idx,2] * d2[*idx,2] * d1[*idx,3] * d2[*idx,3] + d1[*idx,2] * d3[*idx,2] * d1[*idx,0] * d3[*idx,0] + d1[*idx,2] * d3[*idx,2] * d1[*idx,1] * d3[*idx,1] + d1[*idx,2] * d3[*idx,2] * d1[*idx,2] * d3[*idx,2] + d1[*idx,2] * d3[*idx,2] * d1[*idx,3] * d3[*idx,3] + d1[*idx,3] * d1[*idx,3] * d1[*idx,0] * d1[*idx,0] + d1[*idx,3] * d1[*idx,3] * d1[*idx,1] * d1[*idx,1] + d1[*idx,3] * d1[*idx,3] * d1[*idx,2] * d1[*idx,2] + d1[*idx,3] * d1[*idx,3] * d1[*idx,3] * d1[*idx,3] + d1[*idx,3] * d2[*idx,3] * d1[*idx,0] * d2[*idx,0] + d1[*idx,3] * d2[*idx,3] * d1[*idx,1] * d2[*idx,1] + d1[*idx,3] * d2[*idx,3] * d1[*idx,2] * d2[*idx,2] + d1[*idx,3] * d2[*idx,3] * d1[*idx,3] * d2[*idx,3] + d1[*idx,3] * d3[*idx,3] * d1[*idx,0] * d3[*idx,0] + d1[*idx,3] * d3[*idx,3] * d1[*idx,1] * d3[*idx,1] + d1[*idx,3] * d3[*idx,3] * d1[*idx,2] * d3[*idx,2] + d1[*idx,3] * d3[*idx,3] * d1[*idx,3] * d3[*idx,3] + d2[*idx,0] * d1[*idx,0] * d2[*idx,0] * d1[*idx,0] + d2[*idx,0] * d1[*idx,0] * d2[*idx,1] * d1[*idx,1] + d2[*idx,0] * d1[*idx,0] * d2[*idx,2] * d1[*idx,2] + d2[*idx,0] * d1[*idx,0] * d2[*idx,3] * d1[*idx,3] + d2[*idx,0] * d2[*idx,0] * d2[*idx,0] * d2[*idx,0] + d2[*idx,0] * d2[*idx,0] * d2[*idx,1] * d2[*idx,1] + d2[*idx,0] * d2[*idx,0] * d2[*idx,2] * d2[*idx,2] + d2[*idx,0] * d2[*idx,0] * d2[*idx,3] * d2[*idx,3] + d2[*idx,0] * d3[*idx,0] * d2[*idx,0] * d3[*idx,0] + d2[*idx,0] * d3[*idx,0] * d2[*idx,1] * d3[*idx,1] + d2[*idx,0] * d3[*idx,0] * d2[*idx,2] * d3[*idx,2] + d2[*idx,0] * d3[*idx,0] * d2[*idx,3] * d3[*idx,3] + d2[*idx,1] * d1[*idx,1] * d2[*idx,0] * d1[*idx,0] + d2[*idx,1] * d1[*idx,1] * d2[*idx,1] * d1[*idx,1] + d2[*idx,1] * d1[*idx,1] * d2[*idx,2] * d1[*idx,2] + d2[*idx,1] * d1[*idx,1] * d2[*idx,3] * d1[*idx,3] + d2[*idx,1] * d2[*idx,1] * d2[*idx,0] * d2[*idx,0] + d2[*idx,1] * d2[*idx,1] * d2[*idx,1] * d2[*idx,1] + d2[*idx,1] * d2[*idx,1] * d2[*idx,2] * d2[*idx,2] + d2[*idx,1] * d2[*idx,1] * d2[*idx,3] * d2[*idx,3] + d2[*idx,1] * d3[*idx,1] * d2[*idx,0] * d3[*idx,0] + d2[*idx,1] * d3[*idx,1] * d2[*idx,1] * d3[*idx,1] + d2[*idx,1] * d3[*idx,1] * d2[*idx,2] * d3[*idx,2] + d2[*idx,1] * d3[*idx,1] * d2[*idx,3] * d3[*idx,3] + d2[*idx,2] * d1[*idx,2] * d2[*idx,0] * d1[*idx,0] + d2[*idx,2] * d1[*idx,2] * d2[*idx,1] * d1[*idx,1] + d2[*idx,2] * d1[*idx,2] * d2[*idx,2] * d1[*idx,2] + d2[*idx,2] * d1[*idx,2] * d2[*idx,3] * d1[*idx,3] + d2[*idx,2] * d2[*idx,2] * d2[*idx,0] * d2[*idx,0] + d2[*idx,2] * d2[*idx,2] * d2[*idx,1] * d2[*idx,1] + d2[*idx,2] * d2[*idx,2] * d2[*idx,2] * d2[*idx,2] + d2[*idx,2] * d2[*idx,2] * d2[*idx,3] * d2[*idx,3] + d2[*idx,2] * d3[*idx,2] * d2[*idx,0] * d3[*idx,0] + d2[*idx,2] * d3[*idx,2] * d2[*idx,1] * d3[*idx,1] + d2[*idx,2] * d3[*idx,2] * d2[*idx,2] * d3[*idx,2] + d2[*idx,2] * d3[*idx,2] * d2[*idx,3] * d3[*idx,3] + d2[*idx,3] * d1[*idx,3] * d2[*idx,0] * d1[*idx,0] + d2[*idx,3] * d1[*idx,3] * d2[*idx,1] * d1[*idx,1] + d2[*idx,3] * d1[*idx,3] * d2[*idx,2] * d1[*idx,2] + d2[*idx,3] * d1[*idx,3] * d2[*idx,3] * d1[*idx,3] + d2[*idx,3] * d2[*idx,3] * d2[*idx,0] * d2[*idx,0] + d2[*idx,3] * d2[*idx,3] * d2[*idx,1] * d2[*idx,1] + d2[*idx,3] * d2[*idx,3] * d2[*idx,2] * d2[*idx,2] + d2[*idx,3] * d2[*idx,3] * d2[*idx,3] * d2[*idx,3] + d2[*idx,3] * d3[*idx,3] * d2[*idx,0] * d3[*idx,0] + d2[*idx,3] * d3[*idx,3] * d2[*idx,1] * d3[*idx,1] + d2[*idx,3] * d3[*idx,3] * d2[*idx,2] * d3[*idx,2] + d2[*idx,3] * d3[*idx,3] * d2[*idx,3] * d3[*idx,3] + d3[*idx,0] * d1[*idx,0] * d3[*idx,0] * d1[*idx,0] + d3[*idx,0] * d1[*idx,0] * d3[*idx,1] * d1[*idx,1] + d3[*idx,0] * d1[*idx,0] * d3[*idx,2] * d1[*idx,2] + d3[*idx,0] * d1[*idx,0] * d3[*idx,3] * d1[*idx,3] + d3[*idx,0] * d2[*idx,0] * d3[*idx,0] * d2[*idx,0] + d3[*idx,0] * d2[*idx,0] * d3[*idx,1] * d2[*idx,1] + d3[*idx,0] * d2[*idx,0] * d3[*idx,2] * d2[*idx,2] + d3[*idx,0] * d2[*idx,0] * d3[*idx,3] * d2[*idx,3] + d3[*idx,0] * d3[*idx,0] * d3[*idx,0] * d3[*idx,0] + d3[*idx,0] * d3[*idx,0] * d3[*idx,1] * d3[*idx,1] + d3[*idx,0] * d3[*idx,0] * d3[*idx,2] * d3[*idx,2] + d3[*idx,0] * d3[*idx,0] * d3[*idx,3] * d3[*idx,3] + d3[*idx,1] * d1[*idx,1] * d3[*idx,0] * d1[*idx,0] + d3[*idx,1] * d1[*idx,1] * d3[*idx,1] * d1[*idx,1] + d3[*idx,1] * d1[*idx,1] * d3[*idx,2] * d1[*idx,2] + d3[*idx,1] * d1[*idx,1] * d3[*idx,3] * d1[*idx,3] + d3[*idx,1] * d2[*idx,1] * d3[*idx,0] * d2[*idx,0] + d3[*idx,1] * d2[*idx,1] * d3[*idx,1] * d2[*idx,1] + d3[*idx,1] * d2[*idx,1] * d3[*idx,2] * d2[*idx,2] + d3[*idx,1] * d2[*idx,1] * d3[*idx,3] * d2[*idx,3] + d3[*idx,1] * d3[*idx,1] * d3[*idx,0] * d3[*idx,0] + d3[*idx,1] * d3[*idx,1] * d3[*idx,1] * d3[*idx,1] + d3[*idx,1] * d3[*idx,1] * d3[*idx,2] * d3[*idx,2] + d3[*idx,1] * d3[*idx,1] * d3[*idx,3] * d3[*idx,3] + d3[*idx,2] * d1[*idx,2] * d3[*idx,0] * d1[*idx,0] + d3[*idx,2] * d1[*idx,2] * d3[*idx,1] * d1[*idx,1] + d3[*idx,2] * d1[*idx,2] * d3[*idx,2] * d1[*idx,2] + d3[*idx,2] * d1[*idx,2] * d3[*idx,3] * d1[*idx,3] + d3[*idx,2] * d2[*idx,2] * d3[*idx,0] * d2[*idx,0] + d3[*idx,2] * d2[*idx,2] * d3[*idx,1] * d2[*idx,1] + d3[*idx,2] * d2[*idx,2] * d3[*idx,2] * d2[*idx,2] + d3[*idx,2] * d2[*idx,2] * d3[*idx,3] * d2[*idx,3] + d3[*idx,2] * d3[*idx,2] * d3[*idx,0] * d3[*idx,0] + d3[*idx,2] * d3[*idx,2] * d3[*idx,1] * d3[*idx,1] + d3[*idx,2] * d3[*idx,2] * d3[*idx,2] * d3[*idx,2] + d3[*idx,2] * d3[*idx,2] * d3[*idx,3] * d3[*idx,3] + d3[*idx,3] * d1[*idx,3] * d3[*idx,0] * d1[*idx,0] + d3[*idx,3] * d1[*idx,3] * d3[*idx,1] * d1[*idx,1] + d3[*idx,3] * d1[*idx,3] * d3[*idx,2] * d1[*idx,2] + d3[*idx,3] * d1[*idx,3] * d3[*idx,3] * d1[*idx,3] + d3[*idx,3] * d2[*idx,3] * d3[*idx,0] * d2[*idx,0] + d3[*idx,3] * d2[*idx,3] * d3[*idx,1] * d2[*idx,1] + d3[*idx,3] * d2[*idx,3] * d3[*idx,2] * d2[*idx,2] + d3[*idx,3] * d2[*idx,3] * d3[*idx,3] * d2[*idx,3] + d3[*idx,3] * d3[*idx,3] * d3[*idx,0] * d3[*idx,0] + d3[*idx,3] * d3[*idx,3] * d3[*idx,1] * d3[*idx,1] + d3[*idx,3] * d3[*idx,3] * d3[*idx,2] * d3[*idx,2] + d3[*idx,3] * d3[*idx,3] * d3[*idx,3] * d3[*idx,3])
