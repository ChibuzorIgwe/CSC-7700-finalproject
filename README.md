# CSC-7700-finalproject
An AI-powered assistant for Eclipse reservoir simulator documentation using RNN, LSTM, Transformer models, and Retrieval-Augmented Generation (RAG) with Sentence-BERT and FLAN-T5. This project simplifies complex Eclipse keywords and syntax into natural language explanations for petroleum engineers.

# Eclipse AI Assistant

This repository contains the implementation of an AI assistant designed to explain Eclipse reservoir simulator documentation in natural language. It features:

- RNN, LSTM, and Transformer language models trained on domain-specific question-answer pairs.
- A Retrieval-Augmented Generation (RAG) system that combines Sentence-BERT-based chunk retrieval with FLAN-T5 generation.
- Evaluation metrics including BLEU, Perplexity, and prompt-based qualitative analysis.

## Features

- 🔍 Understand Eclipse keywords via natural language queries
- 🧠 Custom-trained RNN, LSTM, and Transformer models
- 📚 Retrieval-augmented generation (RAG) using Sentence-BERT and FLAN-T5
- 📊 Evaluation using BLEU score, perplexity, and visualized loss curves
- 💬 Command-line interface for interactive question-answering

## Tech Stack

- PyTorch
- Hugging Face Transformers
- Sentence-BERT (`all-MiniLM-L6-v2`)
- FLAN-T5 (`google/flan-t5-base`)
- ChromaDB for vector storage
- SentencePiece for BPE tokenization

