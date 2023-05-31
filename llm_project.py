from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm = OpenAI(model_name='text-davinci-003')

keywords_input = input('Enter keywords (ex. "gold, running, butterfly, forest"): ')

# first chain, generate plot
template = """
    Below are a set of keywords.
    Based on the keywords, think of a movie plot.
    The plot should include all the keywords.

    The keywords are: {keywords}
"""
prompt_template = PromptTemplate(
    input_variables = ["keywords"],
    template = template
)
first_chain = LLMChain(llm=llm, prompt=prompt_template)

# second chain, create story
template = """
    Here is the movie plot: {plot}
    Based on the above movie plot, write a story.
    The story should be around 15 sentences long.
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

    Add some details to make the story more exciting to read.
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

print(res)

