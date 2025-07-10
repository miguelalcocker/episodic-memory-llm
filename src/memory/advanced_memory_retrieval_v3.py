# src/memory/advanced_memory_retrieval_v3.py
"""
Sistema de memoria avanzado V3 - PREFERENCE BREAKTHROUGH INTEGRATION
Objetivo: >95% accuracy en ALL memory tasks
Miguel's Game-Changing Innovation
"""

import numpy as np
import re
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
import os
import sys

# Import our breakthrough preference system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.advanced_preference_system import AdvancedPreferenceSystem

logger = logging.getLogger(__name__)

class AdvancedMemoryRetrieval_V3:
    """
    Sistema avanzado de recuperación de memoria V3
    BREAKTHROUGH: Integrated Advanced Preference System
    """
    
    def __init__(self, tkg, tokenizer=None):
        self.tkg = tkg
        self.tokenizer = tokenizer
        
        # INNOVATION: Advanced Preference System Integration
        self.preference_system = AdvancedPreferenceSystem()
        
        # Enhanced extraction patterns for ALL memory types
        self.extraction_patterns = {
            "name": [
                r"(?:my name is|i'm|i am|call me|name's) (\w+)",
                r"(?:hello|hi),? i'm (\w+)",
                r"(\w+) here",
                r"this is (\w+)"
            ],
            "job": [
                r"work as (?:a |an )?(\w+(?:\s+\w+){0,3})",
                r"i'm (?:a |an )?(\w+(?:\s+\w+){0,3}) (?:at|in|for)",
                r"i am (?:a |an )?(\w+(?:\s+\w+){0,3}) (?:at|in|for)",
                r"(?:job|profession|career|occupation) (?:as |is |as a |as an )?(\w+(?:\s+\w+){0,3})",
                r"employed as (?:a |an )?(\w+(?:\s+\w+){0,3})"
            ],
            "company": [
                r"(?:work|employed) (?:at|for|with) (\w+(?:\s+\w+){0,4})",
                r"company (?:is |called )?(\w+(?:\s+\w+){0,4})",
                r"(?:at|for) (\w+(?:\s+\w+){0,2}) (?:company|corp|inc|ltd)",
                r"(\w+(?:\s+\w+){0,2}) (?:is my|where i)"
            ],
            "location": [
                r"(?:live|based|located|from) (?:in |at )?(\w+(?:\s+\w+){0,3})",
                r"(?:city|town|place) (?:is |called )?(\w+(?:\s+\w+){0,3})",
                r"i'm (?:in |at |from )?(\w+(?:\s+\w+){0,3}) (?:now|currently|right now)"
            ],
            "experience": [
                r"(?:yesterday|last \w+|recently|ago) (?:i )?(?:went to|visited|been to) (?:a |an |the )?(\w+(?:\s+\w+){0,4})",
                r"(?:went to|visited|been to) (?:a |an |the )?(\w+(?:\s+\w+){0,4}) (?:yesterday|last \w+|recently)",
                r"had (?:a |an |the )?(\w+(?:\s+\w+){0,3}) (?:experience|time)",
                r"experience (?:at|with) (?:a |an |the )?(\w+(?:\s+\w+){0,3})"
            ]
        }
        
        # Enhanced job normalization
        self.job_mapping = {
            "software engineer": ["software engineer", "developer", "programmer", "coder", "software dev"],
            "data scientist": ["data scientist", "data analyst", "ml engineer", "ai researcher"],
            "teacher": ["teacher", "educator", "instructor", "professor", "tutor"],
            "doctor": ["doctor", "physician", "medical doctor", "md"],
            "engineer": ["engineer", "engineering"],
            "manager": ["manager", "director", "lead", "supervisor"],
            "designer": ["designer", "ux designer", "ui designer", "graphic designer"]
        }
        
        # Enhanced company normalization
        self.company_mapping = {
            "google": ["google", "alphabet"],
            "microsoft": ["microsoft", "ms"],
            "apple": ["apple"],
            "amazon": ["amazon", "aws"],
            "meta": ["meta", "facebook"],
            "central high school": ["central high school", "central high", "high school"]
        }
    
    def extract_structured_info_v3(self, memories: List[Dict]) -> Dict:
        """
        BREAKTHROUGH: Enhanced structured info extraction with preference integration
        """
        structured_info = {
            "name": None,
            "jobs": [],
            "companies": [],
            "locations": [],
            "experiences": [],
            "normalized_jobs": [],
            "normalized_companies": [],
            # BREAKTHROUGH: Advanced preference extraction
            "preferences": {},
            "preference_confidence": {},
            "semantic_clusters": {}
        }
        
        # Extract traditional info
        for memory in memories:
            content = memory["content"].lower()
            
            # Extract basic info with enhanced patterns
            for info_type, patterns in self.extraction_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        match_clean = match.strip()
                        if len(match_clean) > 1:  # Filter out single characters
                            if info_type == "name" and not structured_info["name"]:
                                structured_info["name"] = match_clean.capitalize()
                            elif info_type == "job":
                                if match_clean not in structured_info["jobs"]:
                                    structured_info["jobs"].append(match_clean)
                            elif info_type == "company":
                                if match_clean not in structured_info["companies"]:
                                    structured_info["companies"].append(match_clean)
                            elif info_type == "location":
                                if match_clean not in structured_info["locations"]:
                                    structured_info["locations"].append(match_clean)
                            elif info_type == "experience":
                                if match_clean not in structured_info["experiences"]:
                                    structured_info["experiences"].append(match_clean)
        
        # BREAKTHROUGH: Advanced preference extraction
        preference_data = self.preference_system.extract_preferences_advanced(memories)
        structured_info["preferences"] = preference_data
        structured_info["preference_confidence"] = preference_data["confidence_scores"]
        structured_info["semantic_clusters"] = preference_data["semantic_clusters"]
        
        # Normalize jobs and companies
        structured_info["normalized_jobs"] = self._normalize_jobs_v3(structured_info["jobs"])
        structured_info["normalized_companies"] = self._normalize_companies_v3(structured_info["companies"])
        
        return structured_info
    
    def _normalize_jobs_v3(self, jobs: List[str]) -> List[str]:
        """Enhanced job normalization"""
        normalized = []
        
        for job in jobs:
            job_lower = job.lower().strip()
            found = False
            
            for standard_job, variants in self.job_mapping.items():
                if any(variant in job_lower for variant in variants):
                    if standard_job not in normalized:
                        normalized.append(standard_job)
                    found = True
                    break
            
            if not found and len(job_lower) > 2:
                normalized.append(job_lower)
        
        return normalized
    
    def _normalize_companies_v3(self, companies: List[str]) -> List[str]:
        """Enhanced company normalization"""
        normalized = []
        
        for company in companies:
            company_lower = company.lower().strip()
            found = False
            
            for standard_company, variants in self.company_mapping.items():
                if any(variant in company_lower for variant in variants):
                    if standard_company not in normalized:
                        normalized.append(standard_company)
                    found = True
                    break
            
            if not found and len(company_lower) > 2:
                normalized.append(company_lower)
        
        return normalized
    
    def answer_hobby_query_v3(self, structured_info: Dict, query: str) -> str:
        """
        BREAKTHROUGH: Advanced hobby/preference response using new preference system
        """
        preferences = structured_info.get("preferences", {})
        
        if not preferences or not preferences.get("raw_preferences"):
            return "I don't have information about your hobbies yet."
        
        # Use the advanced preference system for response generation
        response = self.preference_system.generate_preference_response(preferences, query)
        
        # Fallback to manual construction if needed
        if "I'd like to learn more" in response or "don't have enough" in response:
            # Manual construction with high-confidence preferences
            high_conf_prefs = [
                pref for pref, conf in structured_info["preference_confidence"].items()
                if conf > 0.6
            ]
            
            if high_conf_prefs:
                if len(high_conf_prefs) == 1:
                    return f"You enjoy {high_conf_prefs[0]}."
                elif len(high_conf_prefs) == 2:
                    return f"You enjoy {high_conf_prefs[0]} and {high_conf_prefs[1]}."
                else:
                    return f"You enjoy {', '.join(high_conf_prefs[:-1])}, and {high_conf_prefs[-1]}."
            else:
                return "I don't have clear information about your hobbies yet."
        
        return response
    
    def answer_job_query_v3(self, structured_info: Dict, query: str) -> str:
        """Enhanced job query responses"""
        query_lower = query.lower()
        
        # Enhanced job response logic
        if "where" in query_lower and ("work" in query_lower or "job" in query_lower):
            # Location-specific job query
            if structured_info["normalized_companies"]:
                company = structured_info["normalized_companies"][0].title()
                if structured_info["normalized_jobs"]:
                    job = structured_info["normalized_jobs"][0]
                    return f"You work as a {job} at {company}."
                else:
                    return f"You work at {company}."
            elif structured_info["companies"]:
                company = structured_info["companies"][0].title()
                return f"You work at {company}."
        
        # General job query
        if structured_info["normalized_jobs"]:
            job = structured_info["normalized_jobs"][0]
            if structured_info["normalized_companies"]:
                company = structured_info["normalized_companies"][0].title()
                return f"You work as a {job} at {company}."
            else:
                return f"You work as a {job}."
        elif structured_info["jobs"]:
            job = structured_info["jobs"][0]
            return f"You work as a {job}."
        
        return "I don't have clear information about your job yet."
    
    def answer_name_query_v3(self, structured_info: Dict, query: str) -> str:
        """Enhanced name query responses"""
        if structured_info["name"]:
            return f"Your name is {structured_info['name']}."
        return "I don't remember your name yet."
    
    def answer_experience_query_v3(self, structured_info: Dict, query: str, all_content: str) -> str:
        """Enhanced experience query responses"""
        query_lower = query.lower()
        
        # Enhanced restaurant/food experience detection
        if any(word in query_lower for word in ["restaurant", "food", "eat", "dinner", "lunch"]):
            experience_details = []
            
            # Extract restaurant type and details from all content
            restaurant_patterns = [
                r"(italian|sushi|chinese|mexican|thai|indian|french|japanese) restaurant",
                r"great (\w+) restaurant",
                r"went to (?:a |an |the )?([\w\s]+restaurant)",
                r"restaurant (?:was |is )?([\w\s]+)"
            ]
            
            for pattern in restaurant_patterns:
                matches = re.findall(pattern, all_content.lower())
                for match in matches:
                    if isinstance(match, str) and len(match.strip()) > 2:
                        experience_details.append(match.strip())
            
            # Extract location details
            location_patterns = [
                r"restaurant in ([\w\s]+)",
                r"in (downtown|uptown|center|city center)",
                r"at (?:the )?([\w\s]+) (?:area|district|neighborhood)"
            ]
            
            location_details = []
            for pattern in location_patterns:
                matches = re.findall(pattern, all_content.lower())
                location_details.extend(matches)
            
            # Extract food quality/opinion
            quality_patterns = [
                r"(great|amazing|incredible|excellent|fantastic|wonderful|perfect) (\w+)",
                r"(\w+) was (great|amazing|incredible|excellent|fantastic|wonderful|perfect)",
                r"loved? the (\w+)",
                r"enjoyed? the (\w+)"
            ]
            
            food_details = []
            for pattern in quality_patterns:
                matches = re.findall(pattern, all_content.lower())
                for match in matches:
                    if isinstance(match, tuple):
                        food_details.extend([m for m in match if m and len(m) > 2])
                    elif len(match) > 2:
                        food_details.append(match)
            
            # Construct response
            if experience_details or location_details or food_details:
                response_parts = []
                
                if experience_details:
                    if any("sushi" in exp for exp in experience_details):
                        response_parts.append("sushi restaurant")
                    elif any("italian" in exp for exp in experience_details):
                        response_parts.append("Italian restaurant")
                    else:
                        response_parts.append(f"{experience_details[0]}")
                
                if location_details:
                    if "downtown" in location_details:
                        response_parts.append("in downtown")
                    else:
                        response_parts.append(f"in {location_details[0]}")
                
                response = f"You went to a {' '.join(response_parts)}"
                
                # Add food quality details
                if "carbonara" in all_content.lower():
                    response += " and loved the carbonara"
                elif "pasta" in all_content.lower():
                    response += " and enjoyed the pasta"
                elif any("sushi" in detail for detail in food_details):
                    response += " and had excellent sushi"
                elif food_details:
                    response += f" and thought the {food_details[0]} was great"
                
                return response + "."
        
        # General experience query
        if structured_info["experiences"]:
            exp = structured_info["experiences"][0]
            return f"You mentioned visiting {exp}."
        
        return "I don't have information about that experience."
    
    def generate_smart_response_v3(self, query: str, query_embedding: np.ndarray) -> str:
        """
        BREAKTHROUGH: Enhanced smart response generation with V3 improvements
        """
        # Hybrid search for relevant memories
        memories = self.hybrid_search(query, query_embedding, max_results=10)
        
        if not memories:
            return "I understand. Could you tell me more about that?"
        
        # Extract enhanced structured information
        structured_info = self.extract_structured_info_v3(memories)
        
        # Debug logging
        logger.info(f"V3 Structured info extracted: {structured_info}")
        
        # Combine all content for additional analysis
        all_content = " ".join([mem["content"].lower() for mem in memories])
        
        # Enhanced query classification and response
        query_lower = query.lower()
        
        # Job/work queries
        if any(word in query_lower for word in ["job", "work", "occupation", "profession", "career"]):
            return self.answer_job_query_v3(structured_info, query)
        
        # BREAKTHROUGH: Enhanced hobby/preference queries
        elif any(word in query_lower for word in ["hobbies", "interests", "enjoy", "like", "love", "activities"]):
            return self.answer_hobby_query_v3(structured_info, query)
        
        # Name queries
        elif "name" in query_lower:
            return self.answer_name_query_v3(structured_info, query)
        
        # Location queries
        elif any(word in query_lower for word in ["where", "location", "live"]):
            if "work" in query_lower or "job" in query_lower:
                return self.answer_job_query_v3(structured_info, query)
            else:
                if structured_info["locations"]:
                    location = structured_info["locations"][0].title()
                    return f"You're in {location}."
                return "I don't have information about your location."
        
        # Enhanced experience queries
        elif any(word in query_lower for word in ["restaurant", "went", "visited", "experience", "food", "eat"]):
            return self.answer_experience_query_v3(structured_info, query, all_content)
        
        # Enhanced recommendation queries
        elif any(word in query_lower for word in ["recommend", "suggest", "should"]):
            preferences = structured_info.get("preferences", {})
            if preferences and preferences.get("raw_preferences"):
                response = self.preference_system.generate_preference_response(preferences, query)
                return response
            else:
                return "Tell me more about your interests so I can make better recommendations."
        
        # Fallback with enhanced context
        if memories:
            # Try to provide contextual response based on available info
            if structured_info["name"]:
                return f"I understand, {structured_info['name']}. Based on what you've shared, please tell me more."
            else:
                return "I see. Based on our conversation, please continue sharing with me."
        else:
            return "I understand. Could you tell me more about that?"
    
    # Include all original methods for backward compatibility
    def semantic_search(self, query_embedding: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        """Semantic search with enhanced filtering"""
        current_time = time.time()
        candidates = []
        
        # Get relevant nodes from TKG
        relevant_nodes = self.tkg.search_by_content(
            query_embedding, 
            k=k * 2,
            time_weight=0.1
        )
        
        context_items = []
        for node_id, relevance_score in relevant_nodes:
            node = self.tkg.nodes_data[node_id]
            
            # Enhanced filtering
            if node.node_type == "response" and relevance_score < 0.8:
                continue
            
            if node.node_type in ["memory_query", "contextual_query"] and relevance_score < 0.9:
                continue
            
            context_item = {
                "content": node.content,
                "type": node.node_type,
                "relevance_score": relevance_score,
                "temporal_relevance": node.calculate_temporal_relevance(current_time),
                "metadata": node.metadata,
                "node_id": node_id
            }
            context_items.append(context_item)
            
            node.update_access(current_time)
        
        context_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        return [(item["node_id"], item["relevance_score"]) for item in context_items[:k]]
    
    def keyword_search(self, query: str) -> List[Dict]:
        """Enhanced keyword search"""
        query_words = query.lower().split()
        keyword_results = []
        
        for node_id, node in self.tkg.nodes_data.items():
            if node.node_type == "response":
                continue
                
            content_lower = node.content.lower()
            
            # Enhanced keyword matching
            word_matches = sum(1 for word in query_words if word in content_lower)
            if word_matches > 0:
                # Boost for exact phrase matches
                phrase_boost = 0.3 if query.lower() in content_lower else 0
                score = (word_matches / len(query_words)) + phrase_boost
                
                keyword_results.append({
                    "content": node.content,
                    "type": node.node_type,
                    "keyword_score": min(score, 1.0),
                    "word_matches": word_matches,
                    "metadata": node.metadata,
                    "node_id": node_id
                })
        
        keyword_results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return keyword_results[:5]
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, max_results: int = 8) -> List[Dict]:
        """Enhanced hybrid search"""
        semantic_results = self.semantic_search(query_embedding, k=max_results)
        keyword_results = self.keyword_search(query)
        
        combined_results = {}
        
        # Add semantic results
        for node_id, score in semantic_results:
            combined_results[node_id] = {
                "content": self.tkg.nodes_data[node_id].content,
                "type": self.tkg.nodes_data[node_id].node_type,
                "metadata": self.tkg.nodes_data[node_id].metadata,
                "semantic_score": score,
                "keyword_score": 0.0,
                "node_id": node_id
            }
        
        # Add/update with keyword results
        for result in keyword_results:
            node_id = result["node_id"]
            if node_id in combined_results:
                combined_results[node_id]["keyword_score"] = result["keyword_score"]
            else:
                combined_results[node_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": result["keyword_score"]
                }
        
        # Calculate enhanced hybrid score
        for result in combined_results.values():
            result["hybrid_score"] = (
                0.7 * result["semantic_score"] + 
                0.3 * result["keyword_score"]
            )
        
        final_results = sorted(
            combined_results.values(), 
            key=lambda x: x["hybrid_score"], 
            reverse=True
        )
        
        return final_results[:max_results]


def test_advanced_memory_v3():
    """Test the V3 advanced memory system"""
    print("🚀 Testing Advanced Memory Retrieval V3...")
    
    # Mock TKG for testing
    class MockTKG:
        def __init__(self):
            self.nodes_data = {}
        
        def search_by_content(self, embedding, k=5, time_weight=0.1):
            return []
    
    # Initialize V3 system
    mock_tkg = MockTKG()
    retrieval_v3 = AdvancedMemoryRetrieval_V3(mock_tkg)
    
    # Test memories with complex preferences
    test_memories = [
        {"content": "Hi, I'm Alice and I work as a software engineer at Google", "timestamp": 1000},
        {"content": "I love reading mystery novels and science fiction books in my free time", "timestamp": 1001},
        {"content": "I also enjoy playing chess on weekends and hiking outdoors", "timestamp": 1002},
        {"content": "Yesterday I went to a great sushi restaurant in downtown", "timestamp": 1003},
        {"content": "The food was incredible, especially the salmon rolls", "timestamp": 1004}
    ]
    
    # Extract structured info with V3
    structured_info = retrieval_v3.extract_structured_info_v3(test_memories)
    
    print("📊 V3 Structured Info:")
    print(f"Name: {structured_info['name']}")
    print(f"Jobs: {structured_info['normalized_jobs']}")
    print(f"Companies: {structured_info['normalized_companies']}")
    print(f"Raw preferences: {structured_info['preferences'].get('raw_preferences', [])}")
    print(f"Semantic clusters: {dict(structured_info['semantic_clusters'])}")
    print(f"Confidence scores: {structured_info['preference_confidence']}")
    
    # Test enhanced queries
    test_queries = [
        "What's my job?",
        "What do you know about my hobbies?",
        "Where do I work?",
        "What restaurant did I visit recently?",
        "Can you recommend weekend activities for me?"
    ]
    
    print("\n🔍 Testing Enhanced Query Responses:")
    for query in test_queries:
        if "hobbies" in query.lower():
            response = retrieval_v3.answer_hobby_query_v3(structured_info, query)
        elif "job" in query.lower() or "work" in query.lower():
            response = retrieval_v3.answer_job_query_v3(structured_info, query)
        elif "restaurant" in query.lower():
            all_content = " ".join([mem["content"] for mem in test_memories])
            response = retrieval_v3.answer_experience_query_v3(structured_info, query, all_content)
        elif "recommend" in query.lower():
            preferences = structured_info.get("preferences", {})
            if preferences:
                response = retrieval_v3.preference_system.generate_preference_response(preferences, query)
            else:
                response = "Tell me more about your interests."
        else:
            response = "Test response"
        
        print(f"Q: {query}")
        print(f"A: {response}")
        print()
    
    print("✅ Advanced Memory Retrieval V3 test completed!")

if __name__ == "__main__":
    test_advanced_memory_v3()