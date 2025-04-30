prompt_template="""
You are an Expert Interviewer and today you are taked with creating a 
sets of only 10 interview question and answer from the text given below:
------------------
{text}
------------------------
The Question style will be as follows:
1. They should check the core knowledge of the interviewee
2. It should only be theory question and question about code will be askiing for a rough pseudo code/ steps.
3. The answer of the question should not be too big as the interview needs to be well timed.
4. The Questions need to take into account pre-requisites that one should know if they have studied the given document.
5. I just want question
Create questions that will prepare the coders or programmers for the interview make 
sure not to leave anything important.

Questions:
"""

refine_template = ("""
You are an expert at creating practice questions based on coding material and documentation.
Your goal is to help a coder or programmer prepare for a coding test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones but keep the question count to 10 only
(ONLY IF NECESSARY) with some more context below.
-----------
{text}
-----------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
and in the end i dont want anything else just the questions no intro nothing
Just before starting the question answer pair write <start> 
QUESTIONS:
""")

