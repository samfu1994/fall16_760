import pydot

menu = {"a":{"b" : {"a" :"b"}}}
# {u'plas': {'>21.0': {u'mass': {'>21.0': {u'age': {'<=25.0': {u'pedi': {'>0.509': {u'skin': {'<=215.0': {u'preg': {'<=215.0': {u'pres': {'<=215.0': {u'insu': {'<=215.0': {}, '>215.0': {}}}, '>215.0': {}}}, '>215.0': {}}}, '>215.0': {}}}, '<=0.509': {u'skin': {'<=61.0': {u'pres': {'<=61.0': {u'preg': {'>86.0': {}, '<=86.0': {u'insu': {'>86.0': {}, '<=86.0': {}}}}}, '>61.0': {u'insu': {'>110.0': {u'preg': {'>4.0': {}, '<=4.0': {}}}, '<=110.0': {u'preg': {'<=0.0': {}, '>0.0': {}}}}}}}, '>61.0': {}}}}}, '>25.0': {u'pedi': {'<=0.471': {u'insu': {'<=120.0': {u'skin': {'<=0.0': {u'pres': {'>60.0': {u'preg': {'>6.0': {}, '<=6.0': {}}}, '<=60.0': {}}}, '>0.0': {u'pres': {'>108.0': {}, '<=108.0': {u'preg': {'<=5.0': {}, '>5.0': {}}}}}}}, '>120.0': {u'preg': {'>29.0': {}, '<=29.0': {u'pres': {'>29.0': {u'skin': {'>29.0': {}, '<=29.0': {}}}, '<=29.0': {}}}}}}}, '>0.471': {u'preg': {'<=0.0': {}, '>0.0': {u'insu': {'<=70.0': {u'pres': {'<=0.0': {}, '>0.0': {u'skin': {'<=0.0': {}, '>0.0': {}}}}}, '>70.0': {u'skin': {'>29.0': {u'pres': {'<=88.0': {}, '>88.0': {}}}, '<=29.0': {}}}}}}}}}}}, '<=21.0': {u'pres': {'<=25.0': {}, '>25.0': {}}}}}, '<=21.0': {}}}
# memu = 
def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None, cur):
    for k,v in node.iteritems():
        k = str(cur) + "_" + k
        if isinstance(v, dict):
            # We start wth the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            v = str(cur) + "_" + v
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, k+'_'+v)

graph = pydot.Dot(graph_type='graph')
visit(menu, 0)
graph.write_png('example1_graph.png')