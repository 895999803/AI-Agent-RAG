from openai import OpenAI
from src.config import OPENROUTER_API_KEY, LLM_MODEL, TEMPERATURE
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='rag_project/logs/rag_app.log',
    filemode='a',
    encoding='utf-8'
)

def evaluate_answer_quality(question, context, answer, expected_answer=None):
    """
    Evaluate the quality of an answer using prompt-based scoring.
    
    Args:
        question (str): The question asked
        context (list): List of context chunks used for generation
        answer (str): The generated answer to evaluate
        expected_answer (str, optional): Expected answer for comparison
    
    Returns:
        dict: Evaluation results with scores and details
    """
    try:
        full_context = "\n\n".join(context)
        
        # Create evaluation prompt
        evaluation_prompt = f"""Please evaluate the quality of the following answer based on the provided question and context.

Question: {question}

Context: {full_context}

Answer: {answer}

Please provide a score from 1 to 5 for each of the following criteria:

1. Answer Relevance (1-5): How well does the answer address the question and use the provided context?
2. Answer Correctness (1-5): How factually accurate is the answer based on the context? (Only if expected answer is provided)
3. Answer Completeness (1-5): How complete and comprehensive is the answer?
4. Answer Coherence (1-5): How well-structured and coherent is the answer?

Please respond with ONLY a JSON object in the following format:
{{"relevance": X, "correctness": X, "completeness": X, "coherence": X}}

Where X is a number from 1 to 5 for each criterion."""
        
        if expected_answer:
            evaluation_prompt += f"""

Expected Answer: {expected_answer}

Use the expected answer to help evaluate correctness."""
        
        logging.info(f"Evaluating answer for question: {question}")
        logging.debug(f"Evaluation prompt: {evaluation_prompt[:200]}...")

        try:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator that scores answers on a scale of 1-5 based on specific criteria."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=TEMPERATURE
            )
            evaluation_text = response.choices[0].message.content
            logging.info("Answer evaluation completed.")
            
            # Parse the JSON response
            import json
            try:
                evaluation_scores = json.loads(evaluation_text)
                
                # Validate scores are in range 1-5
                for key, score in evaluation_scores.items():
                    if not isinstance(score, (int, float)) or score < 1 or score > 5:
                        raise ValueError(f"Invalid score for {key}: {score}")
                
                # Calculate overall score
                total_score = sum(evaluation_scores.values())
                overall_score = total_score / len(evaluation_scores)
                
                # Round to 1 decimal place
                overall_score = round(overall_score, 1)
                
                evaluation_result = {
                    'overall_score': overall_score,
                    'max_score': 5.0,
                    'metric_scores': evaluation_scores,
                    'evaluation_passed': True,
                    'details': f"Answer quality evaluation completed successfully"
                }
                
                logging.info(f"Answer evaluation completed. Overall score: {overall_score}/5.0")
                return evaluation_result
                
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing evaluation response: {e}")
                logging.error(f"Response text: {evaluation_text}")
                return {
                    'overall_score': 0.0,
                    'max_score': 5.0,
                    'metric_scores': {},
                    'evaluation_passed': False,
                    'details': f"Error parsing evaluation response: {str(e)}"
                }
                
        except Exception as e:
            logging.error(f"Error during answer evaluation: {e}")
            return {
                'overall_score': 0.0,
                'max_score': 5.0,
                'metric_scores': {},
                'evaluation_passed': False,
                'details': f"Error during evaluation: {str(e)}"
            }
            
    except Exception as e:
        logging.error(f"Error during answer evaluation: {e}")
        return {
            'overall_score': 0.0,
            'max_score': 5.0,
            'metric_scores': {},
            'evaluation_passed': False,
            'details': f"Error during evaluation: {str(e)}"
        }

def evaluate_chunk_relevance(question, chunk, task_type="general"):
    """
    Evaluate the relevance of a single chunk to a question.
    
    Args:
        question (str): The question being asked
        chunk (str): The document chunk to evaluate
        task_type (str): Type of task (general, reading, recommendation, etc.)
    
    Returns:
        dict: Evaluation result with score and details
    """
    try:
        # Create evaluation prompt for chunk relevance
        evaluation_prompt = f"""Please evaluate how relevant the following document chunk is to answering the given question.

Question: {question}

Document Chunk: {chunk}

Task Type: {task_type}

Please provide a score from 1 to 10 for the following criteria:

1. Content Relevance (1-10): How directly does the chunk's content relate to the question?
2. Information Completeness (1-10): How much of the information needed to answer the question is present in the chunk?
3. Context Quality (1-10): How well does the chunk provide useful context for answering the question?

Please respond with ONLY a JSON object in the following format:
{{"content_relevance": X, "information_completeness": X, "context_quality": X, "overall_score": X}}

Where X is a number from 1 to 10 for each criterion, and overall_score is the average of all three scores."""
        
        logging.info(f"Evaluating chunk relevance for question: {question[:50]}...")
        logging.debug(f"Chunk content: {chunk[:200]}...")

        try:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator that scores document chunks on their relevance to questions on a scale of 1-10."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=TEMPERATURE
            )
            evaluation_text = response.choices[0].message.content
            logging.info("Chunk relevance evaluation completed.")
            
            # Parse the JSON response
            import json
            try:
                evaluation_scores = json.loads(evaluation_text)
                
                # Validate scores are in range 1-10
                for key, score in evaluation_scores.items():
                    if not isinstance(score, (int, float)) or score < 1 or score > 10:
                        raise ValueError(f"Invalid score for {key}: {score}")
                
                # Ensure overall_score is calculated if not provided
                if 'overall_score' not in evaluation_scores:
                    relevant_scores = [v for k, v in evaluation_scores.items() if k != 'overall_score']
                    evaluation_scores['overall_score'] = sum(relevant_scores) / len(relevant_scores)
                
                # Round to 1 decimal place
                for key in evaluation_scores:
                    evaluation_scores[key] = round(float(evaluation_scores[key]), 1)
                
                evaluation_result = {
                    'chunk_score': evaluation_scores['overall_score'],
                    'max_score': 10.0,
                    'metric_scores': evaluation_scores,
                    'evaluation_passed': True,
                    'details': f"Chunk relevance evaluation completed successfully"
                }
                
                logging.info(f"Chunk evaluation completed. Score: {evaluation_scores['overall_score']}/10.0")
                return evaluation_result
                
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing chunk evaluation response: {e}")
                logging.error(f"Response text: {evaluation_text}")
                return {
                    'chunk_score': 0.0,
                    'max_score': 10.0,
                    'metric_scores': {},
                    'evaluation_passed': False,
                    'details': f"Error parsing evaluation response: {str(e)}"
                }
                
        except Exception as e:
            logging.error(f"Error during chunk evaluation: {e}")
            return {
                'chunk_score': 0.0,
                'max_score': 10.0,
                'metric_scores': {},
                'evaluation_passed': False,
                'details': f"Error during evaluation: {str(e)}"
            }
            
    except Exception as e:
        logging.error(f"Error during chunk evaluation: {e}")
        return {
            'chunk_score': 0.0,
            'max_score': 10.0,
            'metric_scores': {},
            'evaluation_passed': False,
            'details': f"Error during evaluation: {str(e)}"
        }

def evaluate_chunks_relevance(question, chunks, task_type="general"):
    """
    Evaluate relevance of multiple chunks and return them sorted by score.
    
    Args:
        question (str): The question being asked
        chunks (list): List of document chunks to evaluate
        task_type (str): Type of task (general, reading, recommendation, etc.)
    
    Returns:
        list: List of tuples (chunk, score, evaluation_details) sorted by score descending
    """
    try:
        chunk_evaluations = []
        
        for i, chunk in enumerate(chunks):
            logging.info(f"Evaluating chunk {i+1}/{len(chunks)} for relevance")
            evaluation = evaluate_chunk_relevance(question, chunk, task_type)
            
            if evaluation['evaluation_passed']:
                chunk_evaluations.append({
                    'chunk': chunk,
                    'score': evaluation['chunk_score'],
                    'details': evaluation
                })
            else:
                # If evaluation failed, assign a low score
                chunk_evaluations.append({
                    'chunk': chunk,
                    'score': 0.0,
                    'details': evaluation
                })
        
        # Sort by score in descending order
        chunk_evaluations.sort(key=lambda x: x['score'], reverse=True)
        
        logging.info(f"Completed evaluation of {len(chunk_evaluations)} chunks")
        return chunk_evaluations
        
    except Exception as e:
        logging.error(f"Error during batch chunk evaluation: {e}")
        # Return chunks with zero scores if evaluation fails
        return [{
            'chunk': chunk,
            'score': 0.0,
            'details': {'evaluation_passed': False, 'details': str(e)}
        } for chunk in chunks]

def format_evaluation_display(evaluation_result):
    """
    Format evaluation result for display in Streamlit.
    
    Args:
        evaluation_result (dict): Result from evaluate_answer_quality
    
    Returns:
        str: Formatted display string
    """
    if not evaluation_result['evaluation_passed']:
        return f"Evaluation failed: {evaluation_result['details']}"
    
    overall_score = evaluation_result['overall_score']
    max_score = evaluation_result['max_score']
    
    # Create score bar visualization
    filled_bars = int(overall_score)
    empty_bars = 5 - filled_bars
    score_bar = "█" * filled_bars + "░" * empty_bars
    
    # Determine score text (no emojis)
    if overall_score >= 4.5:
        score_text = "Excellent"
    elif overall_score >= 3.5:
        score_text = "Good"
    elif overall_score >= 2.5:
        score_text = "Fair"
    elif overall_score >= 1.5:
        score_text = "Poor"
    else:
        score_text = "Very Poor"
    
    display_text = f"""
    **Answer Quality Score: {overall_score}/{max_score}** ({score_text})
    
    **Score Breakdown:**
    {score_bar}
    """
    
    # Add individual metric scores if available
    if evaluation_result['metric_scores']:
        display_text += "\n**Individual Metrics:**\n"
        for metric_name, score in evaluation_result['metric_scores'].items():
            display_text += f"- {metric_name}: {score}/5.0\n"
    
    return display_text

def format_chunk_evaluation_display(chunk_evaluations):
    """
    Format chunk evaluation results for display in Streamlit.
    
    Args:
        chunk_evaluations (list): List of chunk evaluation results
    
    Returns:
        str: Formatted display string
    """
    if not chunk_evaluations:
        return "No chunks to evaluate."
    
    display_text = "**Document Chunk Relevance Scores:**\n\n"
    
    for i, eval_result in enumerate(chunk_evaluations, 1):
        score = eval_result['score']
        chunk_preview = eval_result['chunk'][:100] + "..." if len(eval_result['chunk']) > 100 else eval_result['chunk']
        
        # Create score bar visualization (scaled to 10 for chunks)
        filled_bars = int(score)
        empty_bars = 10 - filled_bars
        score_bar = "█" * filled_bars + "░" * empty_bars
        
        display_text += f"**Chunk {i}: {score}/10.0**\n"
        display_text += f"{score_bar}\n"
        display_text += f"Content: {chunk_preview}\n\n"
    
    return display_text
