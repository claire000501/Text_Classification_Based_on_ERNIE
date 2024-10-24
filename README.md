# Text_Classification_Based_on_ERNIE
This project involves fine-tuning ERNIE to classify each investor comment.  

ERNIE is a pre-trained language model developed by Baidu, based on the Transformer architecture and an extension of the BERT model. It integrates external knowledge sources such as knowledge graphs to enhance the model’s ability to understand semantics. ERNIE is specifically trained with a large amount of Chinese data and knowledge graphs, making it more capable of understanding Chinese semantics and structure. It handles long-range dependencies and ambiguities in the Chinese language more effectively than models that only rely on word context.  

The main objective of this project is to use this model to determine the trust classification that investors have toward listed companies based on their comments. It tries different controlling parameters like temperature, top_p, and penalty_score to find the best combinations that generates the most accurate classification results.
