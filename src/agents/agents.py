from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# Model Initialization 
llm = ChatGroq(model="openai/gpt-oss-120b" , temperature=0 )


# 1st Agent : Search Agent
def build_search_agent():
    return  create_agent(
        model=llm ,
        tools=[web_search] , 
        system_prompt=''' 

                You are a web research agent.

                        Your job is to answer the user's questions using the web search tool
                        whenever up-to-date, factual, or externally verified information is needed.

                        Rules:
                        - Use the web_search tool when the answer requires current or external information.
                        - Analyze the search results carefully before answering.
                        - If the search results are insufficient, perform another search with a better query.
                        - Do not make up information that is not supported by the search results.
                        - Give a clear and concise final answer.


                '''
    )



# Model Initialization for Reader Agent
reader_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# 2nd Agent : Reader Agent

def build_reader_agent():
    return create_agent(
        model= reader_llm,
        tools=[scrape_url],
        system_prompt='''

                You are a web page reader and information extraction agent.

                        Your job is to read and analyze web pages using the scrape_url tool.

                        Rules:
                        - Use the scrape_url tool to retrieve the content of a given URL.
                        - Carefully analyze the scraped content.
                        - Extract only information that is relevant to the user's research question.
                        - Ignore advertisements, navigation menus, unrelated links, and other irrelevant content.
                        - Do not invent or assume information that is not present in the scraped content.
                        - If the scraped content is insufficient or unclear, state that clearly.
                        - Return the relevant findings in a clear and structured format.
                        - Preserve important facts, numbers, names, dates, and claims accurately.

                '''

    )




#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert research writer.

        Your job is to transform the gathered research into a clear,
        accurate, structured, and professional research report.

        Rules:
        - Use only information supported by the provided research.
        - Do not invent facts, statistics, claims, or sources.
        - Resolve contradictions when possible using the available research.
        - Clearly distinguish facts from interpretations.
        - Include all relevant source URLs provided in the research.
        """
    ),
    (
        "human",
        """
        Write a detailed research report on the topic below.

        Topic:
        {topic}

        Research Gathered:
        {research}

        Structure the report as:

        ## Introduction
        Provide context and explain the topic.

        ## Key Findings
        Present at least 3 important findings.
        Explain each finding clearly and support it with the research.

        ## Conclusion
        Summarize the most important insights.

        ## Sources
        List all URLs that were actually provided in the research.

        Be factual, detailed, and professional.
        Do not add information that is not supported by the research.
        """
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a strict and constructive research critic.

        Your job is to evaluate the quality, accuracy, completeness,
        and reliability of a research report.

        Be objective and specific.

        Pay special attention to:
        - Factual accuracy
        - Unsupported claims
        - Missing important information
        - Logical consistency
        - Quality of explanations
        - Relevance to the topic
        - Source usage
        - Clarity and structure

        Do not praise the report without identifying whether it has
        meaningful weaknesses.
        """
    ),
    (
        "human",
        """
        Review the research report below and evaluate it strictly.

        Report:
        {report}

        Respond in exactly this format:

        Score: X/10

        Strengths:
        - ...
        - ...

        Areas to Improve:
        - ...
        - ...

        One line verdict:
        ...

        Give specific and actionable criticism.
        """
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
