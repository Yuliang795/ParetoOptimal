cimport cython
cimport numpy as np 
import numpy as np

from libc.stdlib cimport malloc, free
# cdef struct IntPair:
#     int x
#     int y

# cdef IntPair* point_list = <IntPair*>malloc(0)


cdef class connected_component:
	cdef list cc_point_list
	cdef list excl_cc_set


	def  __init__(self,init_pair, init_excl=None):
		self.cc_point_list = [*init_pair]
		if init_excl != None:
			self.excl_cc_set=[init_excl]
		else:
			self.excl_cc_set=[]
    
	cpdef void add_excl_enc(self, int encoding):
		self.excl_cc_set.append(encoding)

	cpdef void update_cc(self, int point):
		self.cc_point_list.append(point)

	cpdef void merge_cc(self, connected_component obj_cc):
		self.cc_point_list.extend(obj_cc.cc_point_list)
		self.excl_cc_set.extend(obj_cc.excl_cc_set)

	@cython.boundscheck(False)  # Deactivate bounds checking
	@cython.wraparound(False)   # Deactivate negative indexing.
	cpdef int check_excl(self, connected_component obj_cc):
		if np.intersect1d(np.asarray(self.excl_cc_set,dtype=np.int64),np.asarray(obj_cc.excl_cc_set,dtype=np.int64),assume_unique=True ).size>0:
			return 1
		return 0

	cpdef list get_cc_point_list(self):
		return self.cc_point_list
	
	cpdef list get_excl_cc_set(self):
		return self.excl_cc_set
	
	def print_cc(self):
		for i in self.cc_point_list:
			print(i)




# intersect.pyx




