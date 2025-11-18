"""
Threat Intelligence API Connectors
Phase 2: External Threat Intelligence

Implements specific API connectors for various threat intelligence sources
using the Phase 1 API connector framework.
"""

import json
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from pathlib import Path

from api_connector import (
    APIConfig, APIConnectorFactory, AuthType, RequestMethod,
    BaseAPIConnector, APIKeyConnector, BearerTokenConnector, RateLimit
)
from data_lake import DataLakeManager, DataZone
from data_catalog import DataCatalog, DataAsset, DataSensitivity, DataQuality, DataLineage
from data_quality import DataQualityValidator
import logging

class VirusTotalConnector(APIKeyConnector):
    """
    VirusTotal API connector for malware and URL analysis
    """

    def __init__(self, api_key: str):
        config = APIConfig(
            base_url="https://www.virustotal.com/api/v3",
            auth_type=AuthType.API_KEY,
            auth_credentials={"api_key": api_key, "key_name": "x-apikey"},
            rate_limit=RateLimit(requests_per_minute=4, requests_per_hour=500)  # VT free tier limits
        )
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def scan_url(self, url: str) -> Dict[str, Any]:
        """Scan a URL for threats"""
        # First, submit URL for analysis
        submit_response = self.make_request(
            RequestMethod.POST,
            "/urls",
            data=f"url={url}",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if not submit_response.success:
            return {"error": submit_response.error_message}

        analysis_id = submit_response.data.get("data", {}).get("id")

        if not analysis_id:
            return {"error": "Failed to submit URL for analysis"}

        # Wait a moment for analysis
        time.sleep(2)

        # Get analysis results
        result_response = self.make_request(RequestMethod.GET, f"/analyses/{analysis_id}")

        if not result_response.success:
            return {"error": result_response.error_message}

        return self._parse_virustotal_response(result_response.data, url)

    def scan_file_hash(self, file_hash: str) -> Dict[str, Any]:
        """Get analysis results for a file hash"""
        response = self.make_request(RequestMethod.GET, f"/files/{file_hash}")

        if not response.success:
            return {"error": response.error_message}

        return self._parse_virustotal_response(response.data, file_hash)

    def _parse_virustotal_response(self, data: Dict, target: str) -> Dict[str, Any]:
        """Parse VirusTotal API response"""
        try:
            attributes = data.get("data", {}).get("attributes", {})

            # Extract threat scores
            last_analysis_stats = attributes.get("last_analysis_stats", {})
            malicious = last_analysis_stats.get("malicious", 0)
            suspicious = last_analysis_stats.get("suspicious", 0)
            total_scans = sum(last_analysis_stats.values())

            # Calculate threat score (0-1)
            threat_score = (malicious + suspicious * 0.5) / max(total_scans, 1)

            return {
                "target": target,
                "threat_score": min(threat_score, 1.0),
                "malicious_detections": malicious,
                "suspicious_detections": suspicious,
                "total_scans": total_scans,
                "scan_date": attributes.get("last_analysis_date"),
                "file_type": attributes.get("type_description", ""),
                "file_size": attributes.get("size", 0),
                "tags": attributes.get("tags", []),
                "source": "virustotal"
            }
        except Exception as e:
            self.logger.error(f"Error parsing VT response: {str(e)}")
            return {"error": f"Parse error: {str(e)}"}

class PhishTankConnector(BaseAPIConnector):
    """
    PhishTank API connector for phishing site detection
    """

    def __init__(self, api_key: Optional[str] = None):
        config = APIConfig(
            base_url="https://data.phishtank.com",
            auth_type=AuthType.NONE if not api_key else AuthType.API_KEY,
            auth_credentials={"api_key": api_key} if api_key else {},
            rate_limit=RateLimit(requests_per_minute=10, requests_per_hour=100)
        )
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def check_url(self, url: str) -> Dict[str, Any]:
        """Check if URL is in PhishTank database"""
        response = self.make_request(
            RequestMethod.POST,
            "/data/online-valid.json",
            data=f"url={url}&format=json",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if not response.success:
            return {"error": response.error_message}

        return self._parse_phishtank_response(response.data, url)

    def get_recent_phishes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent phishing submissions"""
        response = self.make_request(RequestMethod.GET, f"/data/online-valid.json?limit={limit}")

        if not response.success:
            self.logger.error(f"Failed to get recent phishes: {response.error_message}")
            return []

        if isinstance(response.data, list):
            return [self._parse_phishtank_entry(entry) for entry in response.data]
        else:
            return []

    def _parse_phishtank_response(self, data: Any, url: str) -> Dict[str, Any]:
        """Parse PhishTank response"""
        try:
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                return self._parse_phishtank_entry(entry)
            else:
                return {
                    "url": url,
                    "is_phishing": False,
                    "confidence": 0.0,
                    "source": "phishtank"
                }
        except Exception as e:
            self.logger.error(f"Error parsing PhishTank response: {str(e)}")
            return {"error": f"Parse error: {str(e)}"}

    def _parse_phishtank_entry(self, entry: Dict) -> Dict[str, Any]:
        """Parse individual PhishTank entry"""
        return {
            "url": entry.get("url", ""),
            "is_phishing": True,
            "phish_id": entry.get("phish_id"),
            "phish_detail_url": entry.get("phish_detail_url"),
            "submission_time": entry.get("submission_time"),
            "verified": entry.get("verified", False),
            "verification_time": entry.get("verification_time"),
            "online": entry.get("online", False),
            "target": entry.get("target", ""),
            "confidence": 1.0 if entry.get("verified") else 0.7,
            "source": "phishtank"
        }

class NVDConnector(BaseAPIConnector):
    """
    National Vulnerability Database (NVD) API connector
    """

    def __init__(self, api_key: Optional[str] = None):
        config = APIConfig(
            base_url="https://services.nvd.nist.gov/rest/json",
            auth_type=AuthType.API_KEY if api_key else AuthType.NONE,
            auth_credentials={"api_key": api_key, "key_name": "apiKey"} if api_key else {},
            rate_limit=RateLimit(requests_per_minute=50, requests_per_hour=1000)  # NVD limits
        )
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def get_recent_vulnerabilities(self, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent vulnerabilities from NVD"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            "pubStartDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "pubEndDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "resultsPerPage": min(limit, 2000)
        }

        response = self.make_request(RequestMethod.GET, "/cves/2.0", params=params)

        if not response.success:
            self.logger.error(f"Failed to get vulnerabilities: {response.error_message}")
            return []

        return self._parse_nvd_response(response.data)

    def get_cve_details(self, cve_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific CVE"""
        response = self.make_request(RequestMethod.GET, f"/cves/2.0?cveId={cve_id}")

        if not response.success:
            return {"error": response.error_message}

        vulnerabilities = self._parse_nvd_response(response.data)
        return vulnerabilities[0] if vulnerabilities else {"error": "CVE not found"}

    def _parse_nvd_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse NVD API response"""
        try:
            vulnerabilities = data.get("vulnerabilities", [])

            parsed_vulns = []
            for vuln in vulnerabilities:
                cve = vuln.get("cve", {})

                # Extract CVSS scores
                metrics = cve.get("metrics", {})
                cvss_v3 = metrics.get("cvssMetricV31", [{}])[0] if metrics.get("cvssMetricV31") else {}
                cvss_v2 = metrics.get("cvssMetricV2", [{}])[0] if metrics.get("cvssMetricV2") else {}

                base_score_v3 = cvss_v3.get("cvssData", {}).get("baseScore", 0)
                base_score_v2 = cvss_v2.get("cvssData", {}).get("baseScore", 0)
                severity = cvss_v3.get("cvssData", {}).get("baseSeverity", "UNKNOWN")

                parsed_vulns.append({
                    "cve_id": cve.get("id", ""),
                    "description": cve.get("descriptions", [{}])[0].get("value", ""),
                    "published_date": cve.get("published"),
                    "last_modified": cve.get("lastModified"),
                    "cvss_v3_score": base_score_v3,
                    "cvss_v2_score": base_score_v2,
                    "severity": severity,
                    "references": [ref.get("url") for ref in cve.get("references", [])],
                    "affected_products": self._extract_affected_products(cve),
                    "source": "nvd"
                })

            return parsed_vulns

        except Exception as e:
            self.logger.error(f"Error parsing NVD response: {str(e)}")
            return []

    def _extract_affected_products(self, cve: Dict) -> List[str]:
        """Extract affected products from CVE data"""
        try:
            configurations = cve.get("configurations", [])
            products = []

            for config in configurations:
                nodes = config.get("nodes", [])
                for node in nodes:
                    cpe_match = node.get("cpeMatch", [])
                    for cpe in cpe_match:
                        cpe23_uri = cpe.get("criteria", "")
                        if cpe23_uri.startswith("cpe:2.3"):
                            # Parse CPE string
                            parts = cpe23_uri.split(":")
                            if len(parts) >= 6:
                                vendor = parts[3]
                                product = parts[4]
                                version = parts[5] if len(parts) > 5 else "*"
                                products.append(f"{vendor}:{product}:{version}")

            return list(set(products))  # Remove duplicates

        except Exception:
            return []

class AbuseIPDBConnector(APIKeyConnector):
    """
    AbuseIPDB API connector for IP reputation checking
    """

    def __init__(self, api_key: str):
        config = APIConfig(
            base_url="https://api.abuseipdb.com/api/v2",
            auth_type=AuthType.API_KEY,
            auth_credentials={"api_key": api_key, "key_name": "Key"},
            rate_limit=RateLimit(requests_per_minute=10, requests_per_hour=1000)
        )
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def check_ip(self, ip_address: str, max_age_days: int = 90) -> Dict[str, Any]:
        """Check IP address reputation"""
        params = {
            "ipAddress": ip_address,
            "maxAgeInDays": max_age_days,
            "verbose": True
        }

        response = self.make_request(RequestMethod.GET, "/check", params=params)

        if not response.success:
            return {"error": response.error_message}

        return self._parse_abuseipdb_response(response.data, ip_address)

    def get_blacklisted_ips(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recently blacklisted IPs"""
        # Note: This is a simplified implementation
        # AbuseIPDB doesn't have a direct "blacklist" endpoint
        # In practice, you'd need to maintain your own blacklist or use their reports
        self.logger.warning("AbuseIPDB doesn't provide direct blacklist access")
        return []

    def _parse_abuseipdb_response(self, data: Dict, ip_address: str) -> Dict[str, Any]:
        """Parse AbuseIPDB response"""
        try:
            abuse_data = data.get("data", {})

            return {
                "ip_address": ip_address,
                "abuse_confidence_score": abuse_data.get("abuseConfidenceScore", 0),
                "total_reports": abuse_data.get("totalReports", 0),
                "last_reported_at": abuse_data.get("lastReportedAt"),
                "isp": abuse_data.get("isp", ""),
                "country_code": abuse_data.get("countryCode", ""),
                "domain": abuse_data.get("domain", ""),
                "is_whitelisted": abuse_data.get("isWhitelisted", False),
                "categories": abuse_data.get("categories", {}),
                "reports": abuse_data.get("reports", []),
                "source": "abuseipdb"
            }

        except Exception as e:
            self.logger.error(f"Error parsing AbuseIPDB response: {str(e)}")
            return {"error": f"Parse error: {str(e)}"}

class MalwareDomainListConnector(BaseAPIConnector):
    """
    Malware Domain List API connector
    """

    def __init__(self):
        config = APIConfig(
            base_url="https://www.malwaredomainlist.com",
            auth_type=AuthType.NONE,
            rate_limit=RateLimit(requests_per_minute=10, requests_per_hour=100)
        )
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def get_recent_domains(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent malware domains"""
        # MDL provides data through their website, not a clean API
        # This is a simplified implementation - in practice you'd scrape or use their export
        response = self.make_request(RequestMethod.GET, "/mdl.php")

        if not response.success:
            self.logger.error(f"Failed to get malware domains: {response.error_message}")
            return []

        # This would require HTML parsing in a real implementation
        # For now, return mock data structure
        return self._parse_mdl_response(response.data)

    def _parse_mdl_response(self, data: Any) -> List[Dict[str, Any]]:
        """Parse Malware Domain List response"""
        # Simplified parsing - real implementation would parse HTML table
        try:
            domains = []
            # Mock data for demonstration
            current_time = datetime.now()

            for i in range(min(10, 50)):  # Mock 10 entries
                domains.append({
                    "domain": f"malicious{i}.example.com",
                    "ip_address": f"192.168.1.{i}",
                    "description": f"Malware distribution site {i}",
                    "country": "Unknown",
                    "detection_date": (current_time - timedelta(hours=i)).isoformat(),
                    "source": "malwaredomainlist"
                })

            return domains

        except Exception as e:
            self.logger.error(f"Error parsing MDL response: {str(e)}")
            return []

class ThreatCollector:
    """
    Orchestrates threat intelligence data collection from multiple sources
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connectors = {}
        self.data_lake = DataLakeManager()
        self.catalog = DataCatalog()
        self.quality_validator = DataQualityValidator()

        # Initialize connectors (with placeholder API keys)
        self._initialize_connectors()

    def _initialize_connectors(self):
        """Initialize threat intelligence connectors"""
        # These would be loaded from configuration in production
        connector_configs = {
            'virustotal': {'class': VirusTotalConnector, 'api_key': 'YOUR_VT_API_KEY'},
            'phishtank': {'class': PhishTankConnector, 'api_key': None},
            'nvd': {'class': NVDConnector, 'api_key': None},
            'abuseipdb': {'class': AbuseIPDBConnector, 'api_key': 'YOUR_ABUSEIPDB_API_KEY'},
            'malwaredomainlist': {'class': MalwareDomainListConnector, 'api_key': None}
        }

        for name, config in connector_configs.items():
            try:
                if config['api_key']:
                    connector = config['class'](config['api_key'])
                else:
                    connector = config['class']()

                self.connectors[name] = connector
                self.logger.info(f"Initialized {name} connector")

            except Exception as e:
                self.logger.warning(f"Failed to initialize {name} connector: {str(e)}")

    def collect_all_threats(self) -> Dict[str, Any]:
        """Collect threat intelligence from all configured sources"""
        results = {}

        # Malware indicators
        if 'virustotal' in self.connectors:
            results['malware'] = self._collect_malware_data()

        # Phishing data
        if 'phishtank' in self.connectors:
            results['phishing'] = self._collect_phishing_data()

        # Vulnerabilities
        if 'nvd' in self.connectors:
            results['vulnerabilities'] = self._collect_vulnerability_data()

        # IP reputation
        if 'abuseipdb' in self.connectors:
            results['ip_reputation'] = self._collect_ip_reputation_data()

        # Malicious domains
        if 'malwaredomainlist' in self.connectors:
            results['malicious_domains'] = self._collect_domain_data()

        return results

    def _collect_malware_data(self) -> Dict[str, Any]:
        """Collect malware indicators from VirusTotal"""
        try:
            connector = self.connectors['virustotal']

            # In a real implementation, you'd have a list of URLs/hashes to check
            # For demo, we'll check some known malicious indicators
            test_indicators = [
                "https://example-malicious.com",
                "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"  # Example hash
            ]

            malware_data = []
            for indicator in test_indicators:
                if indicator.startswith("http"):
                    result = connector.scan_url(indicator)
                else:
                    result = connector.scan_file_hash(indicator)

                if 'error' not in result:
                    malware_data.append(result)

            # Store in data lake
            if malware_data:
                df = pd.DataFrame(malware_data)
                dataset_id = f"virustotal_malware_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                file_path = self.data_lake.store_data(
                    df=df,
                    dataset_id=dataset_id,
                    source='virustotal',
                    data_type='malware_indicators'
                )

                # Register in catalog
                self._register_threat_asset(dataset_id, 'virustotal', 'malware_indicators', df)

                return {
                    'success': True,
                    'records_collected': len(malware_data),
                    'dataset_id': dataset_id
                }

        except Exception as e:
            self.logger.error(f"Error collecting malware data: {str(e)}")

        return {'success': False, 'error': str(e)}

    def _collect_phishing_data(self) -> Dict[str, Any]:
        """Collect phishing data from PhishTank"""
        try:
            connector = self.connectors['phishtank']
            phishing_sites = connector.get_recent_phishes(limit=50)

            if phishing_sites:
                df = pd.DataFrame(phishing_sites)
                dataset_id = f"phishtank_phishing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                file_path = self.data_lake.store_data(
                    df=df,
                    dataset_id=dataset_id,
                    source='phishtank',
                    data_type='phishing_sites'
                )

                self._register_threat_asset(dataset_id, 'phishtank', 'phishing_sites', df)

                return {
                    'success': True,
                    'records_collected': len(phishing_sites),
                    'dataset_id': dataset_id
                }

        except Exception as e:
            self.logger.error(f"Error collecting phishing data: {str(e)}")

        return {'success': False, 'error': str(e)}

    def _collect_vulnerability_data(self) -> Dict[str, Any]:
        """Collect vulnerability data from NVD"""
        try:
            connector = self.connectors['nvd']
            vulnerabilities = connector.get_recent_vulnerabilities(days=7, limit=100)

            if vulnerabilities:
                df = pd.DataFrame(vulnerabilities)
                dataset_id = f"nvd_vulnerabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                file_path = self.data_lake.store_data(
                    df=df,
                    dataset_id=dataset_id,
                    source='nvd',
                    data_type='vulnerabilities'
                )

                self._register_threat_asset(dataset_id, 'nvd', 'vulnerabilities', df)

                return {
                    'success': True,
                    'records_collected': len(vulnerabilities),
                    'dataset_id': dataset_id
                }

        except Exception as e:
            self.logger.error(f"Error collecting vulnerability data: {str(e)}")

        return {'success': False, 'error': str(e)}

    def _collect_ip_reputation_data(self) -> Dict[str, Any]:
        """Collect IP reputation data from AbuseIPDB"""
        try:
            connector = self.connectors['abuseipdb']

            # In practice, you'd check IPs from your logs or known suspicious IPs
            test_ips = ["8.8.8.8", "1.1.1.1"]  # Example IPs

            ip_data = []
            for ip in test_ips:
                result = connector.check_ip(ip)
                if 'error' not in result:
                    ip_data.append(result)

            if ip_data:
                df = pd.DataFrame(ip_data)
                dataset_id = f"abuseipdb_ip_reputation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                file_path = self.data_lake.store_data(
                    df=df,
                    dataset_id=dataset_id,
                    source='abuseipdb',
                    data_type='ip_reputation'
                )

                self._register_threat_asset(dataset_id, 'abuseipdb', 'ip_reputation', df)

                return {
                    'success': True,
                    'records_collected': len(ip_data),
                    'dataset_id': dataset_id
                }

        except Exception as e:
            self.logger.error(f"Error collecting IP reputation data: {str(e)}")

        return {'success': False, 'error': str(e)}

    def _collect_domain_data(self) -> Dict[str, Any]:
        """Collect malicious domain data"""
        try:
            connector = self.connectors['malwaredomainlist']
            domains = connector.get_recent_domains(limit=50)

            if domains:
                df = pd.DataFrame(domains)
                dataset_id = f"mdl_malicious_domains_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                file_path = self.data_lake.store_data(
                    df=df,
                    dataset_id=dataset_id,
                    source='malwaredomainlist',
                    data_type='malicious_domains'
                )

                self._register_threat_asset(dataset_id, 'malwaredomainlist', 'malicious_domains', df)

                return {
                    'success': True,
                    'records_collected': len(domains),
                    'dataset_id': dataset_id
                }

        except Exception as e:
            self.logger.error(f"Error collecting domain data: {str(e)}")

        return {'success': False, 'error': str(e)}

    def _register_threat_asset(self, dataset_id: str, source: str, data_type: str, df: pd.DataFrame):
        """Register threat intelligence asset in catalog"""
        try:
            lineage = DataLineage(
                dataset_id=dataset_id,
                parent_datasets=[],
                transformation_steps=[f'Collected from {source} API'],
                created_by='threat_collector',
                created_at=datetime.now()
            )

            asset = DataAsset(
                asset_id=f"{source}_{dataset_id}",
                name=f"{source.upper()} {data_type.replace('_', ' ').title()}",
                description=f"Threat intelligence data collected from {source}",
                dataset_id=dataset_id,
                source_system=source,
                data_type=data_type,
                format='parquet',
                schema={col: str(dtype) for col, dtype in df.dtypes.items()},
                record_count=len(df),
                file_size_bytes=1024,  # Would calculate actual size
                created_at=datetime.now(),
                updated_at=datetime.now(),
                owner='threat_intelligence_system',
                tags=['threat_intelligence', source, data_type],
                sensitivity=DataSensitivity.CONFIDENTIAL,
                quality_score=0.8,  # Would be calculated
                quality_level=DataQuality.GOOD,
                retention_policy='90_days',
                access_permissions={'read': ['analyst', 'admin'], 'write': ['admin']},
                data_lineage=lineage
            )

            self.catalog.register_asset(asset)

        except Exception as e:
            self.logger.error(f"Error registering threat asset: {str(e)}")

# Convenience functions
def create_threat_collector() -> ThreatCollector:
    """Create a threat intelligence collector"""
    return ThreatCollector()

def collect_all_threat_data() -> Dict[str, Any]:
    """Collect threat intelligence from all sources"""
    collector = ThreatCollector()
    return collector.collect_all_threats()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    collector = ThreatCollector()

    print("🚀 Starting threat intelligence collection...")

    results = collector.collect_all_threats()

    print("\n📊 Collection Results:")
    for source, result in results.items():
        if result.get('success'):
            print(f"✓ {source}: {result['records_collected']} records collected")
        else:
            print(f"✗ {source}: Failed - {result.get('error', 'Unknown error')}")

    print("\n🎯 Threat Intelligence Collection Complete!")
