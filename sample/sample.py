import numpy as np
import random as rd

### r sampling

r_vals = np.arange(1.731, 7.731 + 0.1, 0.1)

### Q sampling

vals = [-1./2 , 1./2]

pr_1 = []
pr_2 = []
pr_3 = []
pr_4 = []

uvw_comb = []
uvw_str = []

for u in vals:
    for v in vals:
        for w in vals:
            ell = 1./(np.sqrt(1 + u**2 + v**2 + w**2))
            
            pr_1.append(list(ell*np.array([1,u,v,w])))
            uvw_comb.append([1,u,v,w])
            uvw_str.append(['1','u','v','w'])
            
            pr_1.append(list(ell*np.array([-1,u,v,w])))
            uvw_comb.append([-1,u,v,w])
            uvw_str.append(['-1','u','v','w'])
            
            pr_2.append(list(ell*np.array([w,1,u,v])))
            uvw_comb.append([w,1,u,v])
            uvw_str.append(['w','1','u','v'])
            
            pr_2.append(list(ell*np.array([w,-1,u,v])))
            uvw_comb.append([w,-1,u,v])
            uvw_str.append(['w','-1','u','v'])
            
            pr_3.append(list(ell*np.array([v,w,1,u])))
            uvw_comb.append([v,w,1,u])
            uvw_str.append(['v','w','1','u'])
            
            pr_3.append(list(ell*np.array([v,w,-1,u])))
            uvw_comb.append([v,w,-1,u])
            uvw_str.append(['v','w','-1','u'])
            
            pr_4.append(list(ell*np.array([u,v,w,1])))
            uvw_comb.append([u,v,w,1])
            uvw_str.append(['u','v','w','1'])
            
            pr_4.append(list(ell*np.array([u,v,w,-1])))
            uvw_comb.append([u,v,w,-1])
            uvw_str.append(['u','v','w','-1'])

q_list = pr_1 + pr_2 + pr_3 + pr_4

Q_pre = [ np.array([[q[0]+q[1]*1j, q[2]+q[3]*1j],[-q[2]+q[3]*1j, q[0]-q[1]*1j]]) for q in q_list]

### find and remove duplicates

def remove_duplicate_matrices(matrices):
    seen = set()
    result = []
    uvw_idx = np.zeros(len(matrices))

    for idx,matrix in enumerate(matrices):
        # Convert the matrix to a tuple of tuples to make it hashable
        matrix_tuple_plus = tuple(map(tuple, matrix))
        matrix_tuple_minus = tuple(map(tuple, -matrix))
        if (matrix_tuple_plus and matrix_tuple_minus) not in seen:
            result.append(matrix)
            seen.add(matrix_tuple_plus)
            uvw_idx[idx] = 1

    return result,uvw_idx

Q_vals,uvw_idx = remove_duplicate_matrices(Q_pre)

# save only non duplicate uvw cominations and values

uvw = []
uvw_str2 = []
for i in range(len(uvw_idx)):
    if uvw_idx[i] == 1:
        uvw.append(uvw_comb[i])
        uvw_str2.append(uvw_str[i])
    else:
        pass

### save data

np.save('/home/velni/phd/w/tfm/py/sample/uvw', np.array(uvw))
np.save('/home/velni/phd/w/tfm/py/sample/uvw_str', np.array(uvw_str2))
np.save('/home/velni/phd/w/tfm/py/sample/uvw_str_full', np.array(uvw_str))
np.save('/home/velni/phd/w/tfm/py/sample/uvw_full', np.array(uvw_comb))

# np.save('/home/velni/phd/w/tfm/py/sample/r', r_vals)
# np.save('/home/velni/phd/w/tfm/py/sample/Q', Q_vals)