# def try_to_change_list_contents(the_list):
#     print 'got', the_list
#     the_list.append('four')
#     print 'changed to', the_list

# outer_list = ['one', 'two', 'three']

# print 'before, outer_list =', outer_list
# try_to_change_list_contents(outer_list)
# print 'after, outer_list =', outer_list


def try_to_change_list_reference(the_list):
    print 'got', the_list
    the_list.append("aaa")
    print 'set to', the_list

outer_list = ['we', 'like', 'proper', 'English']

print 'before, outer_list =', outer_list
try_to_change_list_reference(outer_list)
print 'after, outer_list =', outer_list