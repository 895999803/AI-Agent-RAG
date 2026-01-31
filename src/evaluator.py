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
