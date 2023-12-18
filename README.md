# AI_Shopping_Assistant
Welcome to a retail revolution where artificial intelligence meets your shopping needs! In a world where technology is transforming every aspect of our lives, my project emerges as a beacon of innovation in the realm of shopping assistance. Imagine a virtual shopping companion that understands your preferences, guides your choices, and enhances your overall experience—this is the future I envisioned and brought to life.

Shopping can be overwhelming, with countless options and considerations. Traditional methods of shopping assistance, while helpful, lack the personalized touch needed to truly cater to individual preferences. Enter the era of AI-driven shopping assistants, a domain where my project seeks to shine.

Models and Algorithms: At the core of the approach is RAG, a state-of-the-art language model, Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative Al models with facts fetched from external sources. In other words, it fills a gap in how LLMs work. Under the hood, LLMs are neural networks, typically measured by how many parameters they contain. Meta Al researchers introduced Retrieval Augmented Generation (RAG) to address such knowledge-intensive tasks. RAG combines an information retrieval component with a text generator model.

Document Preparation: Our journey begins with the preparation of documents, where each product's essential details are encapsulated. For every product in our dataset, a unique document is created, incorporating information such as title, description, subcategory, category, gender, prodcategory, and brand. This forms the foundation upon which our language model will operate, ensuring a rich understanding of each product.

Setting the Stage: OpenAI and Embedding With the documents ready, we dive into the realm of language models. We employ the formidable GPT-3.5 Turbo from OpenAI, setting the stage for advanced language understanding. Additionally, we utilize the OpenAIEmbedding to enhance the representation of our textual data, ensuring a comprehensive grasp of the underlying semantics.

Node Parsing for Enhanced Understanding: To facilitate effective information retrieval, we implement a node parser with a meticulous approach to text splitting. This process involves chunking the text, ensuring a balanced distribution for optimal comprehension. This step is crucial for preparing the textual data for input into our language model.

Building the Index The culmination of our efforts takes shape in the creation of a Vector Store Index. This index serves as the bedrock for efficient querying and retrieval of information. The carefully crafted documents, coupled with advanced language understanding and parsing techniques, form an index that is ready to respond to user queries. and of course Querying the Index. Returning matching products: using functions to filter the gender based on user query,by embedding the query and searching for matching products from the dataset using the cosine similarity

Thank you for stopping by my project !
