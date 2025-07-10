# src/memory/advanced_memory_retrieval_v3_fixed.py
"""
Sistema de memoria avanzado V3 - SELF-CONTAINED VERSION
Objetivo: >95% accuracy en ALL memory tasks
Miguel's Game-Changing Innovation - NO EXTERNAL DEPENDENCIES
"""

import numpy as np
import re
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class AdvancedPreferenceSystem:
    """
    Sistema revolucionario de captura y recall de preferencias
    Innovation: Multi-layer preference extraction con temporal weighting
    INTEGRATED VERSION - NO EXTERNAL IMPORTS
    """
    
    def __init__(self):
        # INNOVATION 1: Hierarchical Preference Categories
        self.preference_hierarchy = {
            "hobbies_active": {
                "patterns": [
                    r"love (\w+(?:\s+\w+)*ing)",
                    r"enjoy (\w+(?:\s+\w+)*ing)", 
                    r"like (?:to )?(\w+(?:\s+\w+)*)",
                    r"hobby (?:is |are )?(\w+(?:\s+\w+)*)",
                    r"passionate about (\w+(?:\s+\w+)*)",
                    r"really into (\w+(?:\s+\w+)*)"
                ],
                "keywords": ["hiking", "running", "swimming", "climbing", "cycling", "dancing"],
                "category": "active_sports"
            },
            "hobbies_intellectual": {
                "patterns": [
                    r"love reading (\w+(?:\s+\w+)*)",
                    r"enjoy (?:reading )?(\w+(?:\s+\w+)*novels?)",
                    r"like (?:to read )?(\w+(?:\s+\w+)*books?)",
                    r"read (\w+(?:\s+\w+)*)",
                    r"favorite (?:book|novel|author) (?:is |are )?(\w+(?:\s+\w+)*)"
                ],
                "keywords": ["reading", "books", "novels", "mystery", "science fiction", "fantasy", "biography"],
                "category": "intellectual"
            },
            "hobbies_games": {
                "patterns": [
                    r"(?:love|enjoy|like|play) (\w*chess\w*)",
                    r"(?:love|enjoy|like|play) (\w*game\w*)",
                    r"(?:love|enjoy|like) (?:playing )?(\w+(?:\s+\w+)*games?)",
                    r"hobby (?:is |are )?(\w*chess\w*)"
                ],
                "keywords": ["chess", "board games", "video games", "card games", "puzzle"],
                "category": "strategic_games"
            },
            "hobbies_creative": {
                "patterns": [
                    r"(?:love|enjoy|like) (\w*art\w*)",
                    r"(?:love|enjoy|like) (\w*music\w*)",
                    r"(?:love|enjoy|like) (\w*paint\w*)",
                    r"(?:love|enjoy|like) (\w*writ\w*)"
                ],
                "keywords": ["art", "music", "painting", "writing", "drawing", "photography"],
                "category": "creative"
            }
        }
        
        # INNOVATION 2: Contextual Extraction Patterns
        self.compound_patterns = [
            r"(?:love|enjoy|like) (\w+(?:\s+\w+)*) and (\w+(?:\s+\w+)*)",
            r"(?:love|enjoy|like) (\w+(?:\s+\w+)*), (\w+(?:\s+\w+)*),? and (\w+(?:\s+\w+)*)",
            r"(?:weekends?|free time|spare time) (?:I |i )?(?:love|enjoy|like) (\w+(?:\s+\w+)*)",
            r"in my (?:free time|spare time) (?:I |i )?(?:love|enjoy|like) (\w+(?:\s+\w+)*)",
            r"really (?:love|enjoy|like) (\w+(?:\s+\w+)*)",
            r"absolutely (?:love|enjoy|like) (\w+(?:\s+\w+)*)",
            r"passionate about (\w+(?:\s+\w+)*)"
        ]
        
        # INNOVATION 3: Semantic Similarity Mapping
        self.preference_clusters = {
            "outdoor_activities": ["hiking", "camping", "outdoor activities", "nature", "climbing", "trekking"],
            "reading_books": ["reading", "books", "novels", "literature", "mystery novels", "science fiction"],
            "strategic_thinking": ["chess", "board games", "strategy games", "puzzles"],
            "physical_fitness": ["running", "swimming", "cycling", "gym", "fitness", "sports"],
            "creative_arts": ["art", "painting", "drawing", "music", "writing", "photography"],
            "culinary_interests": ["cooking", "baking", "food", "restaurants", "cuisine"]
        }
        
        # INNOVATION 4: Temporal Preference Tracking
        self.preference_timeline = []
        self.preference_reinforcement = defaultdict(int)
    
    def extract_preferences_advanced(self, memories: List[Dict]) -> Dict:
        """BREAKTHROUGH METHOD: Multi-layer preference extraction"""
        extracted_preferences = {
            "raw_preferences": [],
            "categorized_preferences": defaultdict(list),
            "confidence_scores": {},
            "temporal_preferences": [],
            "compound_preferences": [],
            "semantic_clusters": defaultdict(list)
        }
        
        for memory in memories:
            content = memory["content"].lower()
            timestamp = memory.get("timestamp", 0)
            
            # LAYER 1: Hierarchical Pattern Matching
            for category, config in self.preference_hierarchy.items():
                for pattern in config["patterns"]:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if isinstance(match, tuple):
                            for m in match:
                                if m.strip():
                                    self._process_preference_match(
                                        m.strip(), category, content, 
                                        extracted_preferences, timestamp
                                    )
                        else:
                            if match.strip():
                                self._process_preference_match(
                                    match.strip(), category, content,
                                    extracted_preferences, timestamp
                                )
            
            # LAYER 2: Compound Pattern Extraction
            for pattern in self.compound_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        compound_prefs = [m.strip() for m in match if m.strip()]
                        extracted_preferences["compound_preferences"].extend(compound_prefs)
                        for pref in compound_prefs:
                            extracted_preferences["raw_preferences"].append(pref)
            
            # LAYER 3: Keyword Reinforcement
            for cluster_name, keywords in self.preference_clusters.items():
                for keyword in keywords:
                    if keyword in content:
                        self.preference_reinforcement[keyword] += 1
                        extracted_preferences["semantic_clusters"][cluster_name].append(keyword)
        
        # LAYER 4: Semantic Clustering and Normalization
        extracted_preferences = self._apply_semantic_clustering(extracted_preferences)
        
        # LAYER 5: Confidence Scoring
        extracted_preferences = self._calculate_confidence_scores(extracted_preferences)
        
        return extracted_preferences
    
    def _process_preference_match(self, match: str, category: str, content: str, 
                                extracted_preferences: Dict, timestamp: float):
        """Process and categorize a preference match"""
        cleaned_match = self._clean_preference_text(match)
        
        if cleaned_match and len(cleaned_match) > 2:
            extracted_preferences["raw_preferences"].append(cleaned_match)
            extracted_preferences["categorized_preferences"][category].append(cleaned_match)
            extracted_preferences["temporal_preferences"].append((timestamp, cleaned_match, category))
            
            confidence = self._calculate_context_confidence(cleaned_match, content)
            extracted_preferences["confidence_scores"][cleaned_match] = confidence
    
    def _clean_preference_text(self, text: str) -> str:
        """Clean and normalize preference text"""
        stop_words = {"to", "the", "a", "an", "and", "or", "but", "in", "on", "at", "for", "with"}
        cleaned = " ".join(word for word in text.split() if word.lower() not in stop_words)
        
        if cleaned.endswith('s') and len(cleaned) > 4 and cleaned[:-1] in ["novel", "book", "game"]:
            cleaned = cleaned[:-1]
        
        return cleaned.strip()
    
    def _calculate_context_confidence(self, preference: str, content: str) -> float:
        """Calculate confidence score based on context"""
        confidence = 0.5
        
        strong_indicators = ["love", "absolutely", "really", "passionate", "favorite"]
        for indicator in strong_indicators:
            if indicator in content:
                confidence += 0.2
        
        if any(context in content for context in ["free time", "hobby", "weekend"]):
            confidence += 0.1
        
        preference_mentions = content.count(preference.lower())
        confidence += min(preference_mentions * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    def _apply_semantic_clustering(self, extracted_preferences: Dict) -> Dict:
        """Apply semantic clustering to group related preferences"""
        for cluster_name, cluster_keywords in self.preference_clusters.items():
            cluster_prefs = []
            
            for pref in extracted_preferences["raw_preferences"]:
                pref_lower = pref.lower()
                
                if any(keyword in pref_lower for keyword in cluster_keywords):
                    cluster_prefs.append(pref)
                elif any(self._semantic_similarity(pref_lower, keyword) > 0.7 
                        for keyword in cluster_keywords):
                    cluster_prefs.append(pref)
            
            if cluster_prefs:
                extracted_preferences["semantic_clusters"][cluster_name] = list(set(cluster_prefs))
        
        return extracted_preferences
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity heuristic"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence_scores(self, extracted_preferences: Dict) -> Dict:
        """Calculate final confidence scores for all preferences"""
        preference_counts = Counter(extracted_preferences["raw_preferences"])
        
        for pref, count in preference_counts.items():
            base_confidence = extracted_preferences["confidence_scores"].get(pref, 0.5)
            frequency_boost = min(count * 0.15, 0.4)
            cluster_boost = 0.1 if any(pref in prefs for prefs in 
                                     extracted_preferences["semantic_clusters"].values()) else 0
            
            final_confidence = min(base_confidence + frequency_boost + cluster_boost, 1.0)
            extracted_preferences["confidence_scores"][pref] = final_confidence
        
        return extracted_preferences
    
    def generate_preference_response(self, extracted_preferences: Dict, query: str) -> str:
        """BREAKTHROUGH: Generate intelligent preference-based responses"""
        query_lower = query.lower()
        
        high_conf_prefs = [
            pref for pref, conf in extracted_preferences["confidence_scores"].items()
            if conf > 0.6
        ]
        
        active_clusters = {
            cluster: prefs for cluster, prefs in extracted_preferences["semantic_clusters"].items()
            if prefs
        }
        
        if any(word in query_lower for word in ["hobbies", "interests", "enjoy", "like", "love"]):
            return self._generate_comprehensive_hobby_response(high_conf_prefs, active_clusters)
        elif "recommend" in query_lower or "suggest" in query_lower:
            return self._generate_recommendation_response(high_conf_prefs, active_clusters, query_lower)
        else:
            if high_conf_prefs:
                return f"Based on what you've shared, you enjoy {self._format_preference_list(high_conf_prefs[:3])}."
            else:
                return "I'd like to learn more about your interests and hobbies."
    
    def _generate_comprehensive_hobby_response(self, preferences: List[str], clusters: Dict) -> str:
        """Generate comprehensive hobby response"""
        if not preferences:
            return "I don't have enough information about your hobbies yet."
        
        response_parts = []
        
        if "reading_books" in clusters and clusters["reading_books"]:
            reading_prefs = [p for p in preferences if any(r in p.lower() for r in clusters["reading_books"])]
            if reading_prefs:
                response_parts.append(f"reading (especially {reading_prefs[0]})")
        
        if "strategic_thinking" in clusters and clusters["strategic_thinking"]:
            game_prefs = [p for p in preferences if any(g in p.lower() for g in clusters["strategic_thinking"])]
            if game_prefs:
                response_parts.append(f"strategic games like {game_prefs[0]}")
        
        if "outdoor_activities" in clusters and clusters["outdoor_activities"]:
            outdoor_prefs = [p for p in preferences if any(o in p.lower() for o in clusters["outdoor_activities"])]
            if outdoor_prefs:
                response_parts.append(f"outdoor activities such as {outdoor_prefs[0]}")
        
        remaining_prefs = [p for p in preferences[:3] if not any(p.lower() in part for part in response_parts)]
        response_parts.extend(remaining_prefs)
        
        if len(response_parts) == 1:
            return f"You enjoy {response_parts[0]}."
        elif len(response_parts) == 2:
            return f"You enjoy {response_parts[0]} and {response_parts[1]}."
        elif len(response_parts) >= 3:
            return f"You enjoy {', '.join(response_parts[:-1])}, and {response_parts[-1]}."
        else:
            return "You have several interesting hobbies and activities you enjoy."
    
    def _generate_recommendation_response(self, preferences: List[str], clusters: Dict, query: str) -> str:
        """Generate recommendation based on preferences"""
        if not preferences:
            return "Tell me more about your interests so I can make better recommendations."
        
        if "weekend" in query or "activity" in query:
            relevant_prefs = []
            
            if "outdoor_activities" in clusters:
                relevant_prefs.extend(clusters["outdoor_activities"][:2])
            if "strategic_thinking" in clusters:
                relevant_prefs.extend(clusters["strategic_thinking"][:1])
            
            if relevant_prefs:
                return f"Based on your interests in {self._format_preference_list(relevant_prefs)}, I'd recommend similar activities."
            else:
                return f"Given your interest in {preferences[0]}, I can suggest related activities."
        
        return f"Based on your interests in {self._format_preference_list(preferences[:2])}, I can suggest related activities."
    
    def _format_preference_list(self, preferences: List[str]) -> str:
        """Format list of preferences for natural response"""
        if not preferences:
            return ""
        elif len(preferences) == 1:
            return preferences[0]
        elif len(preferences) == 2:
            return f"{preferences[0]} and {preferences[1]}"
        else:
            return f"{', '.join(preferences[:-1])}, and {preferences[-1]}"


class AdvancedMemoryRetrieval_V3:
    """
    Sistema avanzado de recuperación de memoria V3 - SELF-CONTAINED
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
        """BREAKTHROUGH: Enhanced structured info extraction with preference integration"""
        structured_info = {
            "name": None,
            "jobs": [],
            "companies": [],
            "locations": [],
            "experiences": [],
            "normalized_jobs": [],
            "normalized_companies": [],
            "preferences": {},
            "preference_confidence": {},
            "semantic_clusters": {}
        }
        
        # Extract traditional info
        for memory in memories:
            content = memory["content"].lower()
            
            for info_type, patterns in self.extraction_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        match_clean = match.strip()
                        if len(match_clean) > 1:
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
        """BREAKTHROUGH: Advanced hobby/preference response using new preference system"""
        preferences = structured_info.get("preferences", {})
        
        if not preferences or not preferences.get("raw_preferences"):
            return "I don't have information about your hobbies yet."
        
        # Use the advanced preference system for response generation
        response = self.preference_system.generate_preference_response(preferences, query)
        
        # Fallback to manual construction if needed
        if "I'd like to learn more" in response or "don't have enough" in response:
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
    
    def generate_smart_response_v3(self, query: str, query_embedding: np.ndarray) -> str:
        """BREAKTHROUGH: Enhanced smart response generation with V3 improvements"""
        # Hybrid search for relevant memories
        memories = self.hybrid_search(query, query_embedding, max_results=10)
        
        if not memories:
            return "I understand. Could you tell me more about that?"
        
        # Extract enhanced structured information
        structured_info = self.extract_structured_info_v3(memories)
        
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
            if structured_info["name"]:
                return f"Your name is {structured_info['name']}."
            return "I don't remember your name yet."
        
        # Enhanced recommendation queries
        elif any(word in query_lower for word in ["recommend", "suggest", "should"]):
            preferences = structured_info.get("preferences", {})
            if preferences and preferences.get("raw_preferences"):
                response = self.preference_system.generate_preference_response(preferences, query)
                return response
            else:
                return "Tell me more about your interests so I can make better recommendations."
        
        # Fallback with enhanced context
        if structured_info["name"]:
            return f"I understand, {structured_info['name']}. Based on what you've shared, please tell me more."
        else:
            return "I see. Based on our conversation, please continue sharing with me."
    
    def answer_job_query_v3(self, structured_info: Dict, query: str) -> str:
        """Enhanced job query responses"""
        query_lower = query.lower()
        
        if "where" in query_lower and ("work" in query_lower or "job" in query_lower):
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
    
    # Simplified methods for compatibility
    def semantic_search(self, query_embedding: np.ndarray, k: int = 8) -> List[Tuple[str, float]]:
        """Simplified semantic search"""
        current_time = time.time()
        relevant_nodes = self.tkg.search_by_content(query_embedding, k=k, time_weight=0.1)
        
        context_items = []
        for node_id, relevance_score in relevant_nodes:
            if node_id in self.tkg.nodes_data:
                node = self.tkg.nodes_data[node_id]
                if node.node_type != "response" or relevance_score > 0.8:
                    context_items.append((node_id, relevance_score))
                    node.update_access(current_time)
        
        return context_items[:k]
    
    def keyword_search(self, query: str) -> List[Dict]:
        """Simplified keyword search"""
        query_words = query.lower().split()
        keyword_results = []
        
        for node_id, node in self.tkg.nodes_data.items():
            if node.node_type == "response":
                continue
                
            content_lower = node.content.lower()
            word_matches = sum(1 for word in query_words if word in content_lower)
            
            if word_matches > 0:
                score = word_matches / len(query_words)
                keyword_results.append({
                    "content": node.content,
                    "type": node.node_type,
                    "keyword_score": score,
                    "node_id": node_id
                })
        
        keyword_results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return keyword_results[:5]
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, max_results: int = 8) -> List[Dict]:
        """Simplified hybrid search"""
        semantic_results = self.semantic_search(query_embedding, k=max_results)
        keyword_results = self.keyword_search(query)
        
        combined_results = {}
        
        # Add semantic results
        for node_id, score in semantic_results:
            if node_id in self.tkg.nodes_data:
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
        
        # Calculate hybrid score
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