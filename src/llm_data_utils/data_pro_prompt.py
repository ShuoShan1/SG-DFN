class EaPrompt():
    def relation_description_prompt(self, entity_info):
        prompt = f"""
You are an outstanding professor in the field of knowledge graphs. Your task is to generate accurate and standardized English form of relation information descriptions for entities based on user input.
Specifically, you should optimize the description of relations between entities and replace non-standard relation expressions with standardized generic terms. In addition, it is necessary to supplement certain background knowledge when necessary.

Please follow the following knowledge completion rules:
If the number of triples is 5 or more: Generate a description based solely on the given triples, ensuring clear logic and accurate, standard language.
If the number of triples is less than 5: You should use your own background knowledge to appropriately supplement the important relation of the entity. These supplementary relation must be rigorously verified to ensure the authenticity and reliability of the information.

Requirements:
The output must focus on the task of generating standardized descriptions based on relation triples and must not deviate from the main topic.
The generated description must use standardized, universal terminology, and the language must meet academic and professional standards.
The output must be logically structured, concise and clear. Please do not make it too long.
Ensure the accuracy and reliability of the information, and supplementary background knowledge must be from reliable sources.

Sample output:
Newton, also known as Isaac Newton, was born in Woolsthorpe Manor and was a mathematician and physicist. His major contributions include the law of universal gravitation and calculus.

Entity triple information entered by the user: 

{entity_info}
"""
        return prompt



    def attribute_description_prompt(self, entity_info):
        prompt = f"""
You are an outstanding professor in the field of knowledge graphs. Your task is to generate accurate and standardized English form of attribute information descriptions for entities based on user input.
Specifically, you should optimize the description of attributes between entities and replace non-standard attribute expressions with standardized generic terms. In addition, it is necessary to supplement certain background knowledge when necessary.

Please follow the following knowledge completion rules:
If the number of triples is 5 or more: Generate a description based solely on the given triples, ensuring clear logic and accurate, standard language.
If the number of triples is less than 5: You should use your own background knowledge to appropriately supplement the important attribute of the entity. These supplementary attribute must be rigorously verified to ensure the authenticity and reliability of the information.

Requirements:
The output must focus on the task of generating standardized descriptions based on attribute triples and must not deviate from the main topic.
The generated description must use standardized, universal terminology, and the language must meet academic and professional standards.
The output must be logically structured, concise and clear. Please do not make it too long.
Ensure the accuracy and reliability of the information, and supplementary background knowledge must be from reliable sources.

Sample output:
Newton, also known as Isaac Newton, was born in Woolsthorpe Manor and was a mathematician and physicist. His major contributions include the law of universal gravitation and calculus.

Entity triple information entered by the user: 

{entity_info}
"""
        return prompt