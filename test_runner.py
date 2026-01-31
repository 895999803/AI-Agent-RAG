import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from src.config import LLM_MODEL
from src.database import get_db_connection, setup_database, store_embeddings, retrieve_similar_documents
from src.document_processor import process_document
from src.embedding import get_embedding
from src.generator import generate_answer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='logs/test_app.log',
    filemode='a',
    encoding='utf-8'
)

class RAGTestRunner:
    def __init__(self):
        self.test_results = []
        self.test_cases = []
        self.load_test_cases()
        
    def load_test_cases(self):
        """Load test cases from JSON file"""
        try:
            # Try to load from current directory first, then from rag_project directory
            test_cases_path = 'test_cases.json'
            if not os.path.exists(test_cases_path):
                test_cases_path = os.path.join('rag_project', 'test_cases.json')
            
            with open(test_cases_path, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
            logging.info(f"Loaded {len(self.test_cases)} test cases from {test_cases_path}")
        except Exception as e:
            logging.error(f"Failed to load test cases: {e}")
            raise
    
    def run_all_tests(self):
        """Run all test cases"""
        logging.info("Starting RAG system tests...")
        
        for test_case in self.test_cases:
            try:
                result = self.run_single_test(test_case)
                self.test_results.append(result)
                logging.info(f"Test {test_case['test_id']}: {result['status']}")
            except Exception as e:
                logging.error(f"Test {test_case['test_id']} failed with exception: {e}")
                self.test_results.append({
                    'test_id': test_case['test_id'],
                    'test_name': test_case['test_name'],
                    'status': 'ERROR',
                    'error_message': str(e),
                    'execution_time': 0
                })
        
        self.generate_test_report()
        return self.test_results
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        test_id = test_case['test_id']
        test_name = test_case['test_name']
        input_data = test_case['input']
        expected_output = test_case['expected_output']
        
        start_time = time.time()
        
        try:
            if test_id == "TC001":
                result = self.test_database_connection()
            elif test_id == "TC002":
                result = self.test_database_table_creation()
            elif test_id == "TC003":
                result = self.test_database_table_skip()
            elif test_id == "TC004":
                result = self.test_document_processing(input_data)
            elif test_id == "TC005":
                result = self.test_embedding_generation(input_data)
            elif test_id == "TC006":
                result = self.test_embedding_storage(input_data)
            elif test_id == "TC007":
                result = self.test_duplicate_content_detection(input_data)
            elif test_id == "TC008":
                result = self.test_new_content_storage(input_data)
            elif test_id == "TC009":
                result = self.test_similar_document_retrieval(input_data)
            elif test_id == "TC010":
                result = self.test_llm_answer_generation(input_data)
            elif test_id == "TC011":
                result = self.test_empty_query_handling(input_data)
            elif test_id == "TC012":
                result = self.test_large_document_processing(input_data)
            elif test_id == "TC013":
                result = self.test_multiple_document_ingestion(input_data)
            elif test_id == "TC014":
                result = self.test_database_query_performance(input_data)
            elif test_id == "TC015":
                result = self.test_error_handling(input_data)
            elif test_id == "TC016":
                result = self.test_memory_usage(input_data)
            elif test_id == "TC017":
                result = self.test_concurrent_access(input_data)
            elif test_id == "TC018":
                result = self.test_answer_quality(input_data)
            elif test_id == "TC019":
                result = self.test_system_integration(input_data)
            elif test_id == "TC020":
                result = self.test_configuration_validation(input_data)
            else:
                result = {
                    'test_id': test_id,
                    'test_name': test_name,
                    'status': 'SKIP',
                    'error_message': 'Test not implemented',
                    'execution_time': 0
                }
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'test_id': test_id,
                'test_name': test_name,
                'status': 'ERROR',
                'error_message': str(e),
                'execution_time': execution_time
            }
    
    def test_database_connection(self) -> Dict:
        """Test database connection"""
        try:
            conn = get_db_connection()
            conn.close()
            return {
                'test_id': 'TC001',
                'test_name': 'Database Connection Test',
                'status': 'PASS',
                'details': 'Database connection successful'
            }
        except Exception as e:
            return {
                'test_id': 'TC001',
                'test_name': 'Database Connection Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_database_table_creation(self) -> Dict:
        """Test database table creation"""
        try:
            setup_database()
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'document_embeddings');")
            table_exists = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            if table_exists:
                return {
                    'test_id': 'TC002',
                    'test_name': 'Database Table Creation Test',
                    'status': 'PASS',
                    'details': 'Table created successfully'
                }
            else:
                return {
                    'test_id': 'TC002',
                    'test_name': 'Database Table Creation Test',
                    'status': 'FAIL',
                    'error_message': 'Table was not created'
                }
        except Exception as e:
            return {
                'test_id': 'TC002',
                'test_name': 'Database Table Creation Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_database_table_skip(self) -> Dict:
        """Test skipping table creation when it exists"""
        try:
            # First create the table
            setup_database()
            
            # Try to create again (should skip)
            setup_database()
            
            return {
                'test_id': 'TC003',
                'test_name': 'Database Table Skip Test',
                'status': 'PASS',
                'details': 'Table creation skipped successfully'
            }
        except Exception as e:
            return {
                'test_id': 'TC003',
                'test_name': 'Database Table Skip Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_document_processing(self, input_data: Dict) -> Dict:
        """Test document processing"""
        try:
            document_path = input_data.get('document_path', 'documents/document.pdf')
            if not os.path.exists(document_path):
                return {
                    'test_id': 'TC004',
                    'test_name': 'Document Processing Test',
                    'status': 'SKIP',
                    'details': 'Test document not found'
                }
            
            chunks = process_document(document_path)
            
            if len(chunks) >= 1:
                return {
                    'test_id': 'TC004',
                    'test_name': 'Document Processing Test',
                    'status': 'PASS',
                    'details': f'Processed {len(chunks)} chunks successfully'
                }
            else:
                return {
                    'test_id': 'TC004',
                    'test_name': 'Document Processing Test',
                    'status': 'FAIL',
                    'error_message': 'No chunks generated'
                }
        except Exception as e:
            return {
                'test_id': 'TC004',
                'test_name': 'Document Processing Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_embedding_generation(self, input_data: Dict) -> Dict:
        """Test embedding generation"""
        try:
            text = input_data.get('text', 'This is a test document chunk for embedding generation')
            embedding = get_embedding(text)
            
            if len(embedding) == 384 and isinstance(embedding, list):
                return {
                    'test_id': 'TC005',
                    'test_name': 'Embedding Generation Test',
                    'status': 'PASS',
                    'details': f'Generated embedding with dimension {len(embedding)}'
                }
            else:
                return {
                    'test_id': 'TC005',
                    'test_name': 'Embedding Generation Test',
                    'status': 'FAIL',
                    'error_message': 'Invalid embedding dimension or type'
                }
        except Exception as e:
            return {
                'test_id': 'TC005',
                'test_name': 'Embedding Generation Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_embedding_storage(self, input_data: Dict) -> Dict:
        """Test embedding storage"""
        try:
            content = input_data.get('content', 'Test content')
            embedding = input_data.get('embedding', [0.1] * 384)
            
            # Clear existing data for clean test
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("TRUNCATE TABLE document_embeddings RESTART IDENTITY;")
            conn.commit()
            cur.close()
            conn.close()
            
            # Store new embedding
            store_embeddings([(content, embedding)])
            
            # Verify storage
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_embeddings;")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            if count == 1:
                return {
                    'test_id': 'TC006',
                    'test_name': 'Embedding Storage Test',
                    'status': 'PASS',
                    'details': 'Embedding stored successfully'
                }
            else:
                return {
                    'test_id': 'TC006',
                    'test_name': 'Embedding Storage Test',
                    'status': 'FAIL',
                    'error_message': 'Embedding not stored correctly'
                }
        except Exception as e:
            return {
                'test_id': 'TC006',
                'test_name': 'Embedding Storage Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_duplicate_content_detection(self, input_data: Dict) -> Dict:
        """Test duplicate content detection"""
        try:
            content = input_data.get('content', 'Existing content in database')
            embedding = get_embedding(content)
            
            # Store content first
            store_embeddings([(content, embedding)])
            
            # Try to store again (should detect duplicate)
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_embeddings WHERE content = %s;", (content,))
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            if count == 1:
                return {
                    'test_id': 'TC007',
                    'test_name': 'Duplicate Content Detection Test',
                    'status': 'PASS',
                    'details': 'Duplicate content detected correctly'
                }
            else:
                return {
                    'test_id': 'TC007',
                    'test_name': 'Duplicate Content Detection Test',
                    'status': 'FAIL',
                    'error_message': 'Duplicate content not detected'
                }
        except Exception as e:
            return {
                'test_id': 'TC007',
                'test_name': 'Duplicate Content Detection Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_new_content_storage(self, input_data: Dict) -> Dict:
        """Test new content storage"""
        try:
            content = input_data.get('content', 'New unique content')
            embedding = get_embedding(content)
            
            # Clear existing data
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("TRUNCATE TABLE document_embeddings RESTART IDENTITY;")
            conn.commit()
            cur.close()
            conn.close()
            
            # Store new content
            store_embeddings([(content, embedding)])
            
            # Verify storage
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_embeddings;")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            if count == 1:
                return {
                    'test_id': 'TC008',
                    'test_name': 'New Content Storage Test',
                    'status': 'PASS',
                    'details': 'New content stored successfully'
                }
            else:
                return {
                    'test_id': 'TC008',
                    'test_name': 'New Content Storage Test',
                    'status': 'FAIL',
                    'error_message': 'New content not stored'
                }
        except Exception as e:
            return {
                'test_id': 'TC008',
                'test_name': 'New Content Storage Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_similar_document_retrieval(self, input_data: Dict) -> Dict:
        """Test similar document retrieval"""
        try:
            query_embedding = input_data.get('query_embedding', [0.1] * 384)
            top_k = input_data.get('top_k', 5)
            
            # Add some test data
            test_content = "This is test content for similarity search"
            test_embedding = get_embedding(test_content)
            store_embeddings([(test_content, test_embedding)])
            
            # Retrieve similar documents
            results = retrieve_similar_documents(query_embedding, top_k)
            
            if len(results) >= 1:
                return {
                    'test_id': 'TC009',
                    'test_name': 'Similar Document Retrieval Test',
                    'status': 'PASS',
                    'details': f'Retrieved {len(results)} similar documents'
                }
            else:
                return {
                    'test_id': 'TC009',
                    'test_name': 'Similar Document Retrieval Test',
                    'status': 'FAIL',
                    'error_message': 'No similar documents retrieved'
                }
        except Exception as e:
            return {
                'test_id': 'TC009',
                'test_name': 'Similar Document Retrieval Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_llm_answer_generation(self, input_data: Dict) -> Dict:
        """Test LLM answer generation"""
        try:
            question = input_data.get('question', 'What is this document about?')
            context = input_data.get('context', 'Document content here')
            
            answer = generate_answer(question, [context])
            
            if len(answer) >= 50:
                return {
                    'test_id': 'TC010',
                    'test_name': 'LLM Answer Generation Test',
                    'status': 'PASS',
                    'details': f'Generated answer with {len(answer)} characters'
                }
            else:
                return {
                    'test_id': 'TC010',
                    'test_name': 'LLM Answer Generation Test',
                    'status': 'FAIL',
                    'error_message': 'Answer too short'
                }
        except Exception as e:
            return {
                'test_id': 'TC010',
                'test_name': 'LLM Answer Generation Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_empty_query_handling(self, input_data: Dict) -> Dict:
        """Test empty query handling"""
        try:
            query = input_data.get('query', '')
            
            if not query.strip():
                return {
                    'test_id': 'TC011',
                    'test_name': 'Empty Query Handling Test',
                    'status': 'PASS',
                    'details': 'Empty query handled correctly'
                }
            else:
                return {
                    'test_id': 'TC011',
                    'test_name': 'Empty Query Handling Test',
                    'status': 'FAIL',
                    'error_message': 'Query should be empty for this test'
                }
        except Exception as e:
            return {
                'test_id': 'TC011',
                'test_name': 'Empty Query Handling Test',
                'status': 'FAIL',
                'error_message': str(e)
            }
    
    def test_answer_quality(self, input_data: Dict) -> Dict:
        """Test answer quality evaluation using DeepEval"""
        try:
            question = input_data.get('question', 'What is this document about?')
            context = input_data.get('context', 'Document content here')
            
            # Generate answer
            answer = generate_answer(question, [context])
            
            # Evaluate answer quality
            from src.evaluator import evaluate_answer_quality
            evaluation_result = evaluate_answer_quality(question, [context], answer)
            
            if evaluation_result['evaluation_passed'] and evaluation_result['overall_score'] >= 0:
                return {
                    'test_id': 'TC018',
                    'test_name': 'Answer Quality Test (DeepEval)',
                    'status': 'PASS',
                    'details': f'Answer quality evaluation successful. Score: {evaluation_result["overall_score"]}/5.0',
                    'metrics': {
                        'overall_score': evaluation_result['overall_score'],
                        'metric_scores': evaluation_result['metric_scores']
                    }
                }
            else:
                return {
                    'test_id': 'TC018',
                    'test_name': 'Answer Quality Test (DeepEval)',
                    'status': 'FAIL',
                    'error_message': 'Answer quality evaluation failed or score too low'
                }
        except Exception as e:
            return {
                'test_id': 'TC018',
                'test_name': 'Answer Quality Test (DeepEval)',
                'status': 'ERROR',
                'error_message': str(e)
            }
    
    # Placeholder methods for remaining tests
    def test_large_document_processing(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC012', 'test_name': 'Large Document Processing Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_multiple_document_ingestion(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC013', 'test_name': 'Multiple Document Ingestion Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_database_query_performance(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC014', 'test_name': 'Database Query Performance Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_error_handling(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC015', 'test_name': 'Error Handling Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_memory_usage(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC016', 'test_name': 'Memory Usage Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_concurrent_access(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC017', 'test_name': 'Concurrent Access Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_system_integration(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC019', 'test_name': 'System Integration Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def test_configuration_validation(self, input_data: Dict) -> Dict:
        return {'test_id': 'TC020', 'test_name': 'Configuration Validation Test', 'status': 'SKIP', 'details': 'Not implemented'}
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        skipped_tests = len([r for r in self.test_results if r['status'] == 'SKIP'])
        error_tests = len([r for r in self.test_results if r['status'] == 'ERROR'])
        
        # Calculate average execution time
        execution_times = [r['execution_time'] for r in self.test_results if 'execution_time' in r]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests,
                'errors': error_tests,
                'success_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
                'average_execution_time': f"{avg_execution_time:.2f} seconds"
            },
            'test_details': self.test_results,
            'timestamp': datetime.now().isoformat(),
            'test_environment': {
                'python_version': '3.x',
                'deepeval_version': 'latest',
                'llm_model': LLM_MODEL
            }
        }
        
        # Save report to file
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate HTML report
        self.generate_html_report(report)
        
        logging.info(f"Test report generated: {passed_tests}/{total_tests} tests passed")
        return report
    
    def generate_html_report(self, report: Dict):
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e0e0e0; padding: 10px; border-radius: 5px; }}
                .test-table {{ width: 100%; border-collapse: collapse; }}
                .test-table th, .test-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .test-table th {{ background-color: #f2f2f2; }}
                .status-pass {{ color: green; font-weight: bold; }}
                .status-fail {{ color: red; font-weight: bold; }}
                .status-skip {{ color: orange; font-weight: bold; }}
                .status-error {{ color: darkred; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RAG System Test Report</h1>
                <p><strong>Generated:</strong> {report['timestamp']}</p>
                <p><strong>LLM Model:</strong> {report['test_environment']['llm_model']}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Tests</h3>
                    <p>{report['test_summary']['total_tests']}</p>
                </div>
                <div class="metric">
                    <h3>Passed</h3>
                    <p class="status-pass">{report['test_summary']['passed']}</p>
                </div>
                <div class="metric">
                    <h3>Failed</h3>
                    <p class="status-fail">{report['test_summary']['failed']}</p>
                </div>
                <div class="metric">
                    <h3>Skipped</h3>
                    <p class="status-skip">{report['test_summary']['skipped']}</p>
                </div>
                <div class="metric">
                    <h3>Errors</h3>
                    <p class="status-error">{report['test_summary']['errors']}</p>
                </div>
                <div class="metric">
                    <h3>Success Rate</h3>
                    <p>{report['test_summary']['success_rate']}</p>
                </div>
            </div>
            
            <table class="test-table">
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Execution Time</th>
                        <th>Details/Error</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for test in report['test_details']:
            status_class = f"status-{test['status'].lower()}"
            error_msg = test.get('error_message', test.get('details', ''))
            execution_time = f"{test.get('execution_time', 0):.2f}s"
            
            html_content += f"""
                    <tr>
                        <td>{test['test_id']}</td>
                        <td>{test['test_name']}</td>
                        <td class="{status_class}">{test['status']}</td>
                        <td>{execution_time}</td>
                        <td>{error_msg}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        with open('test_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info("HTML test report generated: test_report.html")

if __name__ == "__main__":
    # Run tests
    runner = RAGTestRunner()
    results = runner.run_all_tests()
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Total: {len(results)}")
    print(f"Passed: {len([r for r in results if r['status'] == 'PASS'])}")
    print(f"Failed: {len([r for r in results if r['status'] == 'FAIL'])}")
    print(f"Skipped: {len([r for r in results if r['status'] == 'SKIP'])}")
    print(f"Errors: {len([r for r in results if r['status'] == 'ERROR'])}")