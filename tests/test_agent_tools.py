"""White-box tool-call assertions using a transparent demo agent."""

def demo_agent(question: str):
    trace = []
    if 'multiply' in question.lower():
        args = {'a': 6, 'b': 12}
        trace.append({'tool': 'calculator', 'args': args})
        return str(args['a'] * args['b']), trace
    trace.append({'tool': 'knowledge_search', 'args': {'query': question}})
    return 'No matching article found', trace


def test_correct_tool_and_arguments():
    answer, trace = demo_agent('Multiply 6 by 12')
    assert answer == '72'
    assert trace == [{'tool': 'calculator', 'args': {'a': 6, 'b': 12}}]
    assert len(trace) == 1


def test_no_unnecessary_calculator_call():
    _, trace = demo_agent('Find the refund policy')
    assert [step['tool'] for step in trace] == ['knowledge_search']
