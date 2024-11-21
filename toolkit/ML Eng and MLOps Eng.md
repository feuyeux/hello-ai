# What is a ML Eng?

<https://www.run.ai/guides/machine-learning-engineering>

## Machine Learning Engineer Roles and Responsibilities

Machine learning engineers have two key roles:

- feeding data into machine learning models,
- and deploying these models in production.

**Data ingestion and preparation** is a complex task. The data might come from a variety of sources, often streaming in real time. It needs to be automatically processed, cleaned and prepared to suit the data format and other requirements of the model.

**Deployment** involves taking a prototype model in a development environment and scaling it out to serve real users. This may require running the model on more powerful hardware, enabling access to it via APIs, and allowing for updates and re-training of the model using new data.

In order to achieve these and related tasks, machine learning engineers perform the following activities in an organization:

- **Analyze big datasets** and then determine the best method to prepare the data for analysis.
- **Ingest source data** into machine learning systems to enable machine learning training.
- Collaborate with other data scientists and **build effective data pipelines**.
- **Build the infrastructure required to deploy** a machine learning model in production.
- **Manage, maintain, scale, and improve** machine learning models already running in production environments.
- Work with common **ML algorithms** and relevant software **libraries**.
- Optimize and tweak ML models according to how they behave in production.
- Communicate with relevant stakeholders and key users to understand business requirements, and also **clearly explain the capabilities of the ML model**.
- **Deploy models** to production, initially as a prototype, and then as an API that can **serve predictions** for end users.
- Provide technical **support to data and product teams**, helping relevant parties use and understand machine learning systems and datasets.

## What are the Skills Required to Become a Machine Learning or Deep Learning Engineer?

Here are some of the essential skills required from machine learning engineers:

- **Linux/Unix—**ML engineers working with clustered data and servers typically use Linux or other variants of Unix, and need good command of the operating system
- **Java, C, C++**—these programming languages are commonly used by ML engineers to parse and prepare data for machine learning algorithms
- **GPUs and CUDA programming** – large scale machine learning models use graphical processing units (GPUs) to accelerate workloads. CUDA is the most common programming interface used by GPUs, with strong support by GPU hardware and deep learning frameworks. CUDA is an essential skill for a machine learning engineer.
- **Applied mathematics—**machine learning experts must have strong math skills. Some important mathematical concepts are linear algebra, probability, statistics, multivariate computation, tensors and matrix multiplication, algorithms and optimization.
- **Data modeling and evaluation—**ML engineers must be proficient at evaluating large amounts of data, planning how to effectively model it, and testing how the final system behaves.
- **Neural network architecture**—a set of algorithms used to learn and perform complex cognitive tasks. It uses a network of virtual neurons, mimicking the human brain.
- **Natural Language Processing (NLP)—**allows machines to perform linguistic tasks with similar performance to humans. Common tools and technologies include Word2vec, recurrent neural networks (RNN), gensim, and Natural Language Toolkit (NLTK).
- **Reinforcement Learning—**a set of algorithms that enable machines to learn complex tasks from repeated experience.
- **Distributed computing**—ML engineers need to master distributed computing, both on-premises and in the cloud, to deal with large amounts of data and distributed computations.
- **Spark and Hadoop**—these technologies are commonly used for processing large-scale data sets in preparation for machine learning jobs.

## What Makes a Successful Machine Learning Engineer?

As you start on your machine learning engineering job path, here are a few things that will make you successful at this role.

- **Solid programming skills** – machine learning engineering is founded on software development skills. Become proficient at languages like Python (used in machine learning frameworks and data science), C++ (used in embedded applications), and Java (used in large enterprise applications). Also learn machine learning specific languages like R and Prolog.
- **Strong mathematical foundation** – machine learning is strongly focused on mathematics. To be a successful machine learning engineer you will need either academic training in mathematics and statistics, or at least advanced high school training. Keep in mind that many ML algorithms are extensions of traditional statistical techniques.  
- **Creativity and problem solving** – machine learning is a new field and you’ll need to be creative to find solutions to problems encountered by your organization. Successful machine learning engineers identify systematic issues and find generalized solutions, rather than hunting down bugs one by one.
- **Understanding of iterative processes** – machine learning is driven by trial and error. Most models will initially not work, and will achieve good results through experimentation and fine tuning. You’ll need to develop determination, and be willing to try something multiple times until you find the right approach. At the same time, you’ll need to be flexible and learn when to walk away from a problem when it cannot be solved efficiently.
- **Develop intuitions** – machine learning is not a deterministic field, and the best machine learning engineers have intuitions about data and models. They can review a large data set, identify patterns, and have a feeling about which algorithm might be right to approach the data.
- **Data management expertise** – machine learning is a lot about managing large, messy datasets. Machine learning algorithms rely on data, and a lot of it, to train and achieve accurate predictions. As a machine learning engineer you will need to be proficient at data exploration tools like Excel, Tableau, and Microsoft Power BI, and learn to build a solid data pipeline that can feed your models.

# What is MLOps Eng?

https://neptune.ai/blog/mlops-engineer

## MLOps Engineer sits …

An MLOps Engineer sits between **Machine Learning** **Engineering**, **Software Engineering**, **Data Engineering**, and **DevOps**, combining good practices from all to enable successful deployment and management of machine learning models in prod environments.

In the ML stack of things, MLOps engineer sits towards the far right end, starting with 

1. **Data Scientist**: who formulate solutions, work with the stakeholders and design data-driven solutions to problems at hand.
2. **ML** **Engineers/Data** **Engineer****:** They work their charm on the analysis and models developed by [Data Scientists](https://neptune.ai/blog/ml-engineer-vs-data-scientist) to more prod-ready versions with good code practices and scalable products.
3. **MLOps** **Engineer**: While there is a fine line b/w them and ML Engineers, MLOps engineers sits on the infrastructure side of things rather than development. They work with ML Engineers to get the pipelines up and running, setting up CI/CD, firewalls, and tracking for the longevity of machine learning models.

## **MLOps Engineers vs. Data Scientists** 

Data Scientists specialize in finding and applying the optimum machine learning model to handle business challenges. They experiment with different algorithms, fine-tune their hyperparameters, and then assess and corroborate their results using a range of standards. 

While Data Scientists drive the development of machine learning models, MLOps Engineers enable their deployment, integration, and ongoing management, bridging the gap between data science and operations to ensure the efficient and effective use of machine learning models in real-world applications.

## **MLOps Engineers vs. Software Engineers**

Software Engineers focus on access control, use data gathering, cross-platform integration, and hosting, encompassing various aspects such as architecture, coding, testing, and debugging. 

While Software Engineer handles the broader software development lifecycle, MLOps Engineer brings their expertise in machine learning and operations to effectively deploy and manage machine learning models within software systems.

## **MLOps vs. Data Engineers**

MLOps Engineers primarily focus on the deployment, management, and monitoring of machine learning models in production, bridging the gap between data science and operations.

On the other hand, Data Engineers specialize in designing, building, and maintaining data pipelines and infrastructure for efficient and reliable data processing and storage. 

While there is overlap in some areas, MLOps Engineers concentrate on ML model deployment and management, while Data Engineers focus on data infrastructure and pipeline development.

## **MLOps Engineers vs.** **ML** **Engineers**

ML Engineers enable model deployment automation to production systems. The amount of automation varies within the organization. They take a data scientist’s model and make it accessible to the software that utilizes it. 

Machine learning models are commonly built, tested, and validated using Jupyter notebooks or script files. However, software developers want machine learning models to be available through callable APIs like REST.

**ML** **engineers may sit on** **[platform teams](https://neptune.ai/blog/ml-platform-guide)** **as well as individual ML** **dev** **teams depending on the size of the company and the requirements of their** **machine learning** **models.** 

Here’s what [Amy Bachir](https://www.linkedin.com/in/amybachir/), Senior MLOps Engineer at [Interos Inc](https://www.interos.ai/)., said when questioned, “Is there any difference b/w an ML Engineer and an MLOps Engineer?”

Yes! Absolutely! In my opinion, ML Engineers build and retrain machine learning models. MLOps Engineers enable the ML Engineers. MLOps Engineers build and maintain a platform to enable the development and deployment of machine learning models. They typically do that through standardization, automation, and monitoring. MLOps Engineers reiterate the platform and processes to make the machine learning model development and deployment quicker, more reliable, reproducible, and efficient.

![MLOps-data-professionals](https://i0.wp.com/neptune.ai/wp-content/uploads/2023/08/MLOps-data-professionals.png?resize=1020%2C1020&ssl=1)

## MLOps Engineer job

On a very broader scale, a BAU(Business As Usual 按照惯例) for an MLOps Engineer would be something like:

- Checking deployment pipelines for machine learning models.
- Review Code changes and pull requests from the data science team.
- Triggers CI/CD pipelines after code approvals.
- Monitors pipelines and ensures all tests pass and model artifacts are generated/stored correctly.
- Deploys updated models to prod after pipeline completion.
- Works closely with the software engineering and DevOps team to ensure smooth integration.
- Containerize models using Docker and deploy on cloud platforms (like AWS/GCP/Azure).
- Set up monitoring tools to track various metrics like response time, error rates, and resource utilization.
- Establish alerts and notifications to quickly detect anomalies or deviations from expected behavior.
- Analyze monitoring data, log, files, and system metrics.
- Collaborate with the data science team to develop updated pipelines to cover any faults.
- Documenting and troubleshoots, changes, and optimization.

## **MLOps** **Engineer** **job responsibilities**

- Deploying and operationalizing MLOps, in particular, implementing:
- Model hyperparameter optimization
- Model evaluation and explainability
- Model training and automated retraining
- Model workflows from onboarding, operations to decommissioning
- Model version tracking & governance
- Data archival & version management
- Model and drift monitoring
- Creating and using benchmarks, metrics, and monitoring to measure and improve services.
- Providing best practices and executing POC for automated and efficient model operations at scale.
- Designing and developing scalable MLOps frameworks to support models based on client requirements.
- Being the MLOps expert for the sales team, providing technical design solutions to support RFPs.