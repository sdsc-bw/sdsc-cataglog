[![Documentation](https://img.shields.io/badge/Documentation-Click%20Here-blue?style=flat-square)](sdsc-cataglog/readme.de)

# Topic and Background
The main purpose of this project is to implement an application that can output github tools that correspond to the user input, in the case of the current challenge faced, and that help to deal with this challenge.

We all know that the now popular ChatGPT has shown a strong knowledge base in various fields. Especially when combined with bing search, it is able to provide corresponding websites directly on demand. But ChatGPT has some limitations:
1. ChatGPT knowledge is trained on data before September 2021, so there is no way to provide information after that date.
2. ChatGPT has no way to analyze complex logical relationships.
3. ChatGPT cannot list cited sources, and its reliability is based on the reliability of the source information, which may be inherently wrong, inconsistent, or incorrect or contradictory after being combined by ChatGPT.

# idea
To address the problems of ChatGPT, such as lack of ability to analyze complex input, complex responses, inability to provide real-time tools, and possible errors in the links provided. Our core idea is to:
1. decompose the requirements and only ask simple questions to ChatGPT at a time
2. restrict the output so that the output is brief and linked to the topic
3. use github api to get the latest github, to ensure the popularity and effectiveness of the tool

# run
```
python sdsc_cataglog.py
```

# Example output on the topic of cycling  safety
<img src="sdsc-cataglog/images/output_example.png" alt="output example" width="800" height="600">
