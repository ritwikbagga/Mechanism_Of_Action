# Mechanism Of Action
What is the Mechanism of Action (MoA) of a drug? And why is it important?

In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.

How do we determine the MoAs of a new drug? Model 

One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression or cell viability patterns of drugs with known MoAs.


# model Description
After preprocessing the data and plotting the data/outputs and analysis visually we gauged,
Feedforward neural network made the most sense here as the MoA problem we have is
supervised learning in cases where the data to be learned is neither sequential nor
time-dependent.
We tried multiple models with various depths and number of parameters for the neural network.
We found that the model with three hidden layers (256, 256, 64 neurons with a learning rate of
0.01 and weight-decay 0.00002) performed the best.
We used 3 different kinds of optimizers for the best neural network we found. Adam, AdamW,
and Adamax. We also use a learning rate scheduler to improve the neural network.

# Evaluation 

For every sig_id you will be predicting the probability that the sample had a positive response for each <MoA> target. For \(N\) sig_id rows and \(M\) <MoA> targets, you will be making \(N \times M\) predictions. Submissions are scored by the log loss:

$$ \text{score} = - \frac{1}{M}\sum_{m=1}^{M} \frac{1}{N} \sum_{i=1}^{N} \left[ y_{i,m} \log(\hat{y}_{i,m}) + (1 - y_{i,m}) \log(1 - \hat{y}_{i,m})\right] $$

where:

\(N\) is the number of sig_id observations in the test data (\(i=1,…,N\))
\(M\) is the number of scored MoA targets (\(m=1,…,M\))
\( \hat{y}_{i,m} \) is the predicted probability of a positive MoA response for a sig_id
\( y_{i,m} \) is the ground truth, 1 for a positive response, 0 otherwise
\( log() \) is the natural (base e) logarithm


