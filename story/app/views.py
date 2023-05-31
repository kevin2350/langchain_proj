from django.shortcuts import render
from django.template import Context, Template
from django.views.decorators.csrf import csrf_exempt

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# views
@csrf_exempt
def index(request):
    keywords_input = request.POST.get('keywords')

    res = keywords_input
    if res is not None:
        res = generate(keywords_input)
    else:
        res = ''
    return render(request, 'app/index.html', {'res': res})

# llm
def generate(keywords_input):
    llm = OpenAI(model_name='text-davinci-003', temperature=.8, openai_api_key='')

    # first chain, generate plot
    template = """
        Below are a set of keywords.
        Based on the keywords, think of a book plot.
        Make sure the plot includes every keyword in some aspect.

        The keywords are: {keywords}
    """
    prompt_template = PromptTemplate(
        input_variables = ["keywords"],
        template = template
    )
    first_chain = LLMChain(llm=llm, prompt=prompt_template)

    # second chain, create story
    template = """
        Here is the book plot: {plot}
        Based on the above book plot, write a story.
        Make sure the story is at least 10 sentences long.
        Make sure the story includes an exposition, climax, conflict, and conclusion.   
    """
    prompt_template = PromptTemplate(
        input_variables = ["plot"],
        template = template
    )
    second_chain = LLMChain(llm=llm, prompt=prompt_template)
    second_chain_seq = SimpleSequentialChain(
        chains=[first_chain, second_chain], verbose=False
    )

    # third chain, details
    template = """
        Here is the above story: {story}

        Add a few details to make the story more exciting to read.
    """
    prompt_template = PromptTemplate(
        input_variables = ["story"],
        template=template
    )
    third_chain = LLMChain(llm=llm, prompt=prompt_template)
    third_chain_seq = SimpleSequentialChain(
        chains=[first_chain, second_chain, third_chain], verbose=False
    )

    res = third_chain_seq.run(keywords_input)
    return res
