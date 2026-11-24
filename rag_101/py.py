from ast import parse, get_source_segment


code = """
1 + 2
"""
if not i :
    tree = parse(code, mode='eval')
    print(tree.body._fields)
# cursor = 0
# for node in tree.body:
#     segment = get_source_segment(code, node)
#     node_start = code.find(segment, cursor)
#     node_end = node_start + len(segment)
#    cursor = node_end
#     print( f"\n\n{segment}\nstart:{node_start}\nend:{node_end}\n\n")

