:author: Laura Norén
:email: laura.noren@nyu.edu
:institution: New York University 

:author: Anthony Suen
:email: anthonysuen@berkeley.edu
:institution: University of California, Berkeley

------------------------------------------------
Equity, Scalability, and Sustainability of Data Science Infrastructure
------------------------------------------------

.. class:: abstract

   We seek to understand the current state of equity, scalability, and sustainability of data science education infrastructure in both the U.S. and Canada. Our analysis of the technological, funding, and organizational structure of four types of institutions reveals major divergences in the ability of universities across the United States to provide students with accessible data science education infrastructure, primarily Jupyterhub. We observe that generally liberal arts colleges and other institutions with limited IT staff (e.g. community colleges) have much more difficulty setting up Jupyterhub, compared to well funded private institutions or large public universities with a deep technical bench of IT staff. However by leveraging existing public-private partnerships and the experience of Canada’s national Jupyterhub (Syzygy), the U.S. has an opportunity to provide access to a wider range of institutions and students to the Jupyterhub. 


.. class:: keywords

   data science education, Jupyter, Jupyterhub, higher education

Introduction
------------

Data science education has experienced great demand over the past five years, with increasing numbers of undergraduate programs and majors being developed. This demand has fueled the growth of Jupyterhubs, which creates on demand, cloud based Jupyter notebooks for students and researchers. Jupyterhubs offer many benefits in addition to the flexibility and compatibility of the Jupyter environment, such as customization for different use cases, authentication, and campus-wide computing and storage. Overall, universities have found that utilizing Jupyterhubs improves accessibility and scale existing courses. But what has been the experience been like for universities who have not yet or are attempting to deploy Jupyterhub? 

This paper aims to understand how Jupyterhub is affecting the equity, scalability, and sustainability of data science education by providing four cases studies of how Jupyterhubs are being deployed in academic institutions across the United States and Canada.  These cases include: large, technical universities such as UC Berkeley, liberal arts colleges, private universities with large endownments like Harvard, and the Canadian National Jupyterhub Model. We seek to ask specifically understand what are the barriers in certain institutions that limit their access to Jupyterhub? How many institutions, courses and students can each model of Jupyterhub support? And finally, what are the costs of deploying and maintain Jupyterhub infrastructure with each model? 

We conducted over a dozen qualitative interviews with university faculty and IT staff from around the U.S. and Canada.  We also reviewed documentation found on Github. We structured our analysis by first trying to understand educational goals of institution and how it drives funding' decision/structure. We then delve into the infrastructural costs, capabilities, along with team size. We last measure educational impact, such as the number of students served and the number of classes. We conclude with a summary of the findings and potential ways to improve equity, scalability, and sustainability of the infrastructure. Jupyterhubs provide a solution to these demands. 


Case Study 1: UC Berkeley
------------

In Spring 2015, UC Berkeley became one of the first universities to adopt Jupyterhub. Initially set up for 100 students in the new Foundations of Data Science Course, the Jupyterhub instance, led by Jessica Hamrick, quickly expanded as the data science initiative grew at Berkeley. As of Spring 2018, over 1,000 students take Data 8 each semester and around 3,000 students in Berkeley’s Data Science connectors, modules, and upper division courses utilize the Jupyterhub. An additional 45,000 students utilize the Jupyterhub in a free online EdX version.  UC Berkeley aims to serve large portions of its 30,000 undergraduates with data science tools, thus creating the motivation for it to build the largest Jupyterhub deployments in the country. Its cross campus pedagogical vision is assisted by the presence of a large, internal technical team with many members of the core Jupyter team and strong relations and partnerships with cloud vendors like Microsoft and Google.
				
This infrastructural team is in charge of running Berkeley’s instance of Jupyterhub, “Datahub”, consists of the Dean of the new Data Science Division, one tenured teaching faculty, one full-time staff member, ~10 postdocs and graduate students who can help troubleshoot–many of which are from the core Jupyter team–along with a large, technically proficient undergraduate support staff. The internal technical team which allows UC Berkeley’s work to stand out is that their dev/ops team is competent and is able to work closely with IT in various departments. The Berkeley Datahub also has a well built out workflow with unique features like interactive links and Ok.py for large scale autograding.Long running industry partnerships allow Berkeley to subsist on cloud credits from Microsoft and Google, which provide the infrastructure for Datahub.
		
Despite some of this success, there are several long term issues with the Berkeley model. Replicability of UC Berkeley’s model are not easy given the team composition or industry relations for computing credits. Staffing which supports this venture are shared by many undergraduates, graduate students and postdocs on a temporary basis. The staff generally move on and have other priorities to help advance their careers. Furthermore, relying on postdocs and graduate students is precarious project management; they typically do not advance their careers by doing SysAdmin work. The reliance on free cloud credits can be precarious if there is no internal campus solution. 


Case Study 2: Small Liberal Arts University	
------------

The team interviewed several small liberal arts colleges to see how they have fitted the data science or computer science curriculum by utilizing Jupyterhub. We learned that lack of funding, technical resources (relationship with industry), and a lack of time seem to be major hurdles to a successful deployment of Jupyterhub. At liberal arts colleges, deployments are usually designed for small classes consisting of ~20-30 students and deployed and maintained by one or two professors. There exists little IT help for the professor, as compared to the vast number of support staff at institutions like UC Berkeley. Some smaller institutions have even asked public institutions like UC Berkeley for support. The lack of proper guidance and departmental resources, along with overburdened faculty may dissuade efforts to set up a Jupyterhubs all together. Paying for such technology is also tough and ad hoc in smaller institutions. One of the exceptions is Bryn Mawr. It’s Jupyterhub deployment allows access to a wide range of courses, from introductory computer science courses to computational biology [1]_.

Case Study 3: Wealthy Private Universities	
------------

The conditions are different at well funded private universities. Their ecosystem has IT professional surrounded by other IT professionals. Even if internal IT staff have limitations, rich private universities pay third-party vendor to deploy Jupyterhubs and all related support infrastructure. Harvard has said that “we hired a firm to help us implement Jupyterhub in Amazon AWS cloud”. Compared to smaller liberal arts colleges, the experience is relatively free of frustration since the university covers all costs. Harvard has even been able to reduce costs too. Moving from using a Docker instance per student to using an EC2 instance, bringing costs for small classes from $15 per student to $3 per student. With EC2 its cost ranges from min $34 - max $717/month for 20 users [2]_. 

Most of the classes that have deployed Jupyterhub are still relatively small with most being 12-50 students. At Harvard, JupyterHub was deployed on AWS for two classes in School of Engineering that provided significant customization. The Signal Processing class used Docker-based JupyterHub, where each user provisioned with a docker container notebook. For the Decision Theory class, JupyterHub using a dedicated EC2 instance per user’s notebook, providing better scalability, reliability and cost efficiency. 
Wealthier universities often have multiple JupyterHub deployments. For example, Harvard Medical School runs its own Jupyterhub deployment, while its School of Engineering and Applied Science (SEAS) runs a separate JupyterHub deployment also. 

Despite this hassle-free experience, it runs into major issues in replicability and scalability. Other private universities have less experience with cloud computing or cannot rely on their university’s operating budget to support this type of teaching expense especially if classes were still relatively small (12-50 students). When they scale, costs will grow with each instance, especially when factoring in deployment costs of contractors. These includes work beyond just potential costs for the cloud provider, but hiring outside consultants to setup what the open source solution Jupyterhub. Setting up these individual hub that creates nominally the same type of service is wasteful, and long term costs could potentially sky rocket if there is no unified plan. One potential solution is to adopt a Berkeley’s strategy of utilizing Kubernetes that allows the Jupyterhub to host thousands of students across many courses. 


Case Study 4: Canadian Federation (PIMS)	
------------

In 2017, an initiative in Canada led by the Pacific Institute of Mathematics and Sciences (PIMS) and Compute Canada started a new federated model for Jupyterhub that provides access to numerous institutions across Canada [3]_. This model was also built around the belief that private partners could not be relied upon. Currently, it is the only federated JupyterHub model in existence, supporting more than 15 institutions. The Syzgy platform is run and supported by one full-time system network manager based at PIMS and works with Compute Canada. This System Network manager is in charge of installations; any Canadian University simply ask Compute Canada for a JupyterHub computing allocation and a new cluster will be set up. The system manager is paid for by Compute Canada, and further grants from Canadian federal government ($4.5m) and Alberta ($1m) support professors and teachers. There is also  time donation from professors at 10 different institutions. 
											
There are some potential bottlenecks such as having only one person conducting core management. Some scaling issues also exist as any institution’s Jupyterhub is not able to handle ~2 classes of students concurrently (around 200-300 students). Nonetheless, this is still the most functional model in terms of scale and sustainability. It is further able to accommodate small classes, modules, and also high schools. Funding is a hurdle, not a wall. Teachers can focus on course development while also fostering better cross university collaboration, by sharing experiences and course modules shared on a common network. 

Conclusion  - A Path Forward to a National Jupyterhub 				
------------

We believe that while the grassroots efforts in the U.S. have sparked significant innovation in the realm of data science education infrastructure, it has also created a growing chasm of capabilities between institutions. Increasing training in statistics, computing, and data science is crucial to building the STEM workforce and such a national imperative requires a new model scalably support many small institutions. A centralized model can coexist with the existing grassroot models, providing access to smaller institutions while also creating a more cohesive community to share infrastructure/pedagogical practices. Based on the experiences from these four case studies, we conclude that it is a national imperative to support the development of a National Jupyterhub in order to mitigate the challenges of equity, scalability, and sustainability that currently exist in the grassroots efforts. 

Today, only large public or private universities in the U.S. can provide Jupyterhub for large number of undergraduates. The costs of creating single instance for a single class is non trivial, with IT talent capable of deploying and maintaining Jupyterhubs in high demand. At smaller, resource-constrained institutions, deploying Jupyterhub and integrating it with their work for a single class might be too daunting for one instructor or their university IT staff.  If there is no alternative way to access a Jupyterhub for data science education, smaller, less wealthy institutions and underrepresented communities risk getting left out.

As seen in Canada, a national hub can reduce the burden that individual universities experience today in deployment and maintenance costs. We can refine Canada’s model by deploying of one single large scale Jupyterhub that can support over several thousand students across many institutions. This can happen today by scaffolding onto the NSF National Big Data Hubs and the cloud credits they from partners like Microsoft [4]_. This is a hybrid version of UC Berkeley’s experience with single large scale Jupyterhub deployments and Canada’s Jupyterhub model. 

References
----------
.. [1] Shapiro, J. (2017). Computing Through Biology with Jupyter. https://github.com/BrynMawrCollege/TIDES/blob/master/JupyterDayPhilly/JAShapiro_JupyterDayPhilly_2017-05-19.pdf
.. [2] Harvard. (2018). cloudJHub. https://github.com/harvard/cloudJHub
.. [3] Pacific Institute for the Mathematical Sciences. (2017). Canadians Land on Jupyter. https://www.pims.math.ca/news/canadians-land-jupyter
.. [4] Mandava, V. (2017). NSF Big Data Innovation Hubs collaboration — looking back after one year. https://www.microsoft.com/en-us/research/blog/nsf-big-data-innovation-hubs-collaboration/