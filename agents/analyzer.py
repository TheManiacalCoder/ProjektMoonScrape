import os
import json
import aiohttp
import asyncio
from config.manager import ConfigManager
from pathlib import Path
from colorama import Fore, Style
from typing import List
import re
from datetime import datetime

class OpenRouterAnalyzer:
    def __init__(self, db):
        self.config = ConfigManager()
        self.db = db
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        self.analysis_folder = Path("analysis")
        self.analysis_folder.mkdir(exist_ok=True)
        self.user_prompt = None
        self.benchmark_data = {
            'start_time': None,
            'end_time': None,
            'epochs': [],
            'total_requests': 0,
            'total_tokens': 0
        }

    def set_prompt(self, prompt: str):
        self.user_prompt = prompt

    async def analyze_urls(self, filtered_content: dict):
        try:
            self.benchmark_data['start_time'] = datetime.now()
            print(f"{Fore.CYAN}Performing comprehensive analysis...{Style.RESET_ALL}")
            
            if "final_summary" not in filtered_content:
                raise ValueError("Expected final summary data")
                
            if not self.user_prompt:
                raise ValueError("User prompt not set")
                
            summary = filtered_content["final_summary"]
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Extract and sort URLs
            urls = self._sort_urls_by_relevance(summary)
            most_relevant_url = urls[0] if urls else "No URL found"
            
            best_analysis = None
            best_score = 0.0
            
            for epoch in range(1, 6):
                epoch_start = datetime.now()
                print(f"\n{Fore.CYAN}Starting Epoch {epoch} analysis...{Style.RESET_ALL}")
                
                prompt = f"""
                Analysis Epoch: {epoch}
                
                Directly answer this question: {self.user_prompt}
                Answer this question in a (1) paragraph format.
                
                Use this content as your source:
                {summary}
                
                Most relevant URL: {most_relevant_url}
                
                Requirements:
                - Verify information is current as of {current_date}
                """
                
                payload = {
                    "model": self.config.ai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1 + (epoch * 0.05),
                    "max_tokens": 3000
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            analysis = data['choices'][0]['message']['content']
                            
                            # Calculate score first
                            score = self._evaluate_analysis_quality(analysis, epoch)
                            
                            # Then track benchmark data
                            self.benchmark_data['total_requests'] += 1
                            self.benchmark_data['total_tokens'] += len(analysis.split())
                            epoch_end = datetime.now()
                            epoch_duration = (epoch_end - epoch_start).total_seconds()
                            
                            self.benchmark_data['epochs'].append({
                                'epoch': epoch,
                                'duration': epoch_duration,
                                'score': score,
                                'tokens': len(analysis.split())
                            })
                            
                            if score > best_score:
                                best_analysis = analysis
                                best_score = score
                                print(f"{Fore.GREEN}New best analysis found! Score: {best_score:.2f}{Style.RESET_ALL}")
                                print(f"\n{Fore.CYAN}Best Analysis Preview:{Style.RESET_ALL}")
                                print(analysis[:800] + "...")
                            
                            # Add early exit condition
                            if best_score >= 1.0:
                                print(f"{Fore.GREEN}Reached maximum quality score, moving to next phase{Style.RESET_ALL}")
                                break
                            
                            print(f"Epoch {epoch} analysis preview:")
                            print(analysis[:800] + "...")
                            
                        else:
                            error = await response.text()
                            print(f"{Fore.RED}Epoch {epoch} failed: {error}{Style.RESET_ALL}")
            
            # Save benchmark report only at the end
            if best_analysis:
                await self._save_benchmark_report()
                print(f"\n{Fore.GREEN}Final analysis complete! Best score: {best_score:.2f}{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}Final Analysis:{Style.RESET_ALL}")
                print(best_analysis)
                return best_analysis
            else:
                print(f"{Fore.RED}Failed to generate valid analysis{Style.RESET_ALL}")
                return None
        except Exception as e:
            print(f"{Fore.RED}Error during analysis: {e}{Style.RESET_ALL}")
            return None

    def _sort_urls_by_relevance(self, content: str) -> List[str]:
        # Extract URLs from content
        urls = re.findall(r'https?://[^\s]+', content)
        
        # Score URLs based on relevance factors
        scored_urls = []
        for url in urls:
            score = 0
            
            # Higher score for main domain mentions
            domain = re.sub(r'https?://(www\.)?', '', url)
            domain = re.sub(r'\/.*', '', domain)
            score += content.lower().count(domain) * 0.1
            
            # Higher score for exact URL mentions
            score += content.lower().count(url) * 0.2
            
            # Higher score for earlier mentions
            position = content.lower().find(url)
            if position != -1:
                score += (1 - (position / len(content))) * 0.3
                
            # Higher score for authoritative domains
            if any(auth in domain for auth in ['.gov', '.edu', '.org']):
                score += 0.2
                
            scored_urls.append((url, score))
        
        # Sort by score descending
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return [url for url, score in scored_urls]

    def _get_content_for_url(self, url):
        # Implement content retrieval from your database or storage
        pass

    async def save_report(self, report):
        report_path = self.analysis_folder / "aggregated_analysis.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"{Fore.GREEN}Aggregated report saved to {report_path}{Style.RESET_ALL}")

    def _evaluate_analysis_quality(self, analysis: str, epoch: int) -> float:
        score = 0.0
        
        if analysis:
            score += 0.2
            
        current_year = datetime.now().year
        if str(current_year) in analysis:
            score += 0.1 + (0.02 * epoch)
            
        if "as of" in analysis.lower() or "current" in analysis.lower():
            score += 0.1
            
        structure_components = [
            "### Executive Summary",
            "### Key Findings",
            "### Detailed Analysis",
            "### Recommendations",
            "### Sources"
        ]
        for i, component in enumerate(structure_components):
            if component in analysis:
                score += 0.1 + (0.02 * epoch)
                
        if epoch == 1 and "facts" in analysis.lower():
            score += 0.1
        if epoch == 2 and "evidence" in analysis.lower():
            score += 0.1
        if epoch == 3 and "patterns" in analysis.lower():
            score += 0.1
        if epoch == 4 and "insights" in analysis.lower():
            score += 0.1
        if epoch == 5 and "recommendations" in analysis.lower():
            score += 0.1
            
        score += min(len(analysis) / (2000 + (epoch * 200)), 0.2)
        
        if "clearly" in analysis.lower() or "concisely" in analysis.lower():
            score += 0.05 * epoch
            
        depth_indicators = ["detailed", "in-depth", "comprehensive", "thorough"]
        for indicator in depth_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        if "specific" in analysis.lower() or "precise" in analysis.lower():
            score += 0.05 * epoch
            
        evidence_indicators = ["data", "statistics", "research", "study", "source"]
        for indicator in evidence_indicators:
            if indicator in analysis.lower():
                score += 0.05 * epoch
                
        if "actionable" in analysis.lower() or "recommendation" in analysis.lower():
            score += 0.05 * epoch
            
        return min(score, 1.0)

    async def _save_benchmark_report(self):
        self.benchmark_data['end_time'] = datetime.now()
        total_duration = (self.benchmark_data['end_time'] - self.benchmark_data['start_time']).total_seconds()
        
        report = f"""
        Benchmark Report
        ================
        
        Analysis Overview:
        - Start Time: {self.benchmark_data['start_time']}
        - End Time: {self.benchmark_data['end_time']}
        - Total Duration: {total_duration:.2f} seconds
        - Total Requests: {self.benchmark_data['total_requests']}
        - Total Tokens Processed: {self.benchmark_data['total_tokens']}
        
        Epoch Performance:
        """
        
        for epoch in self.benchmark_data['epochs']:
            report += f"""
            - Epoch {epoch['epoch']}:
              Duration: {epoch['duration']:.2f} seconds
              Score: {epoch['score']:.2f}
              Tokens: {epoch['tokens']}
              TPS: {epoch['tokens']/epoch['duration']:.2f} tokens/second
            """
        
        report_path = Path("benchmark_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"{Fore.GREEN}Benchmark report saved to {report_path}{Style.RESET_ALL}")

async def main(urls):
    analyzer = OpenRouterAnalyzer()
    report = await analyzer.analyze_urls(urls)
    if report:
        await analyzer.save_report(report)

# Example usage
if __name__ == "__main__":
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5"
    ]
    asyncio.run(main(urls)) 