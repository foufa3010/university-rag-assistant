from eval.metrics import token_f1,recall_at_k
def test_recall():
    assert recall_at_k(["a","b"],["b"],2)==1.0
    assert recall_at_k(["a"],["b"],1)==0.0
def test_f1():
    assert token_f1("Students pass CS201","Students pass CS201")==1.0
    assert token_f1("wrong answer","different rule")==0.0
