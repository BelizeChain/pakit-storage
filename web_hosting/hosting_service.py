"""
HTTP Hosting Service for BNS domains.

Serves websites from DAG storage backend.
"""

import logging
import asyncio
from typing import Optional, Dict
from pathlib import Path
from aiohttp import web
import json

logger = logging.getLogger(__name__)


class HostingService:
    """
    HTTP server for BNS domain hosting.
    
    Features:
    - Domain-based routing (Host header)
    - DAG storage backend
    - Bandwidth tracking
    - Subscription verification
    - Custom 404/error pages
    - File serving with caching
    
    Architecture:
    - Receives HTTP request with Host: domain.bz
    - Looks up domain in WebsiteManager
    - Fetches content from DAG storage
    - Tracks bandwidth usage
    - Returns content to client
    """
    
    def __init__(
        self,
        website_manager,
        dag_backend,
        host: str = "127.0.0.1",
        port: int = 8000
    ):
        """
        Initialize hosting service.
        
        Args:
            website_manager: WebsiteManager instance
            dag_backend: DAG storage backend
            host: Listen address (use 0.0.0.0 for Docker/cloud, set via PAKIT_HOST env var)
            port: Listen port
        """
        self.manager = website_manager
        self.dag = dag_backend
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        
        # Request cache: (domain, path) -> (content, content_type, size)
        self.cache: Dict[tuple, tuple] = {}
        self.cache_max_size = 100  # Cache up to 100 files
        
        logger.info(f"HostingService initialized on {host}:{port} (DAG backend)")
    
    async def start(self):
        """Start HTTP server."""
        self.app = web.Application()
        self.app.router.add_route('*', '/{path:.*}', self.handle_request)
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        logger.info(f"🌐 Hosting service running at http://{self.host}:{self.port}")
        logger.info("Ready to serve BNS domains!")
    
    async def stop(self):
        """Stop HTTP server."""
        if self.runner:
            await self.runner.cleanup()
        logger.info("Hosting service stopped")
    
    async def handle_request(self, request: web.Request) -> web.Response:
        """
        Handle incoming HTTP request.
        
        Args:
            request: aiohttp request object
        
        Returns:
            HTTP response with website content or error
        """
        # Extract domain from Host header
        host_header = request.headers.get('Host', '').split(':')[0]  # Remove port
        path = request.match_info.get('path', 'index.html')
        
        # Normalize path
        if not path or path == '/':
            path = 'index.html'
        elif path.startswith('/'):
            path = path[1:]
        
        logger.debug(f"Request: {host_header}/{path}")
        
        # Check cache first
        cache_key = (host_header, path)
        if cache_key in self.cache:
            content, content_type, size = self.cache[cache_key]
            await self.manager.track_bandwidth(host_header, size)
            return web.Response(body=content, content_type=content_type)
        
        # Verify hosting subscription
        if not await self.manager.verify_hosting_subscription(host_header):
            return web.Response(
                text=self._generate_error_page(
                    host_header,
                    "Hosting Subscription Expired",
                    "This domain's hosting subscription has expired. Please renew to continue."
                ),
                content_type='text/html',
                status=402  # Payment Required
            )
        
        # Get content hash for domain
        content_hash = await self.manager.get_domain_content(host_header)
        if not content_hash:
            return web.Response(
                text=self._generate_error_page(
                    host_header,
                    "Domain Not Found",
                    f"No hosting content found for {host_header}"
                ),
                content_type='text/html',
                status=404
            )
        
        # Fetch content from DAG storage
        try:
            content, content_type = await self._fetch_from_dag(content_hash, path)
            
            # Track bandwidth
            size = len(content)
            await self.manager.track_bandwidth(host_header, size)
            
            # Update cache (if not too large)
            if size < 1_000_000 and len(self.cache) < self.cache_max_size:  # 1MB limit
                self.cache[cache_key] = (content, content_type, size)
            
            return web.Response(body=content, content_type=content_type)
            
        except FileNotFoundError:
            return web.Response(
                text=self._generate_error_page(
                    host_header,
                    "404 Not Found",
                    f"File not found: {path}"
                ),
                content_type='text/html',
                status=404
            )
        except Exception as e:
            logger.error(f"Error serving {host_header}/{path}: {e}")
            return web.Response(
                text=self._generate_error_page(
                    host_header,
                    "500 Internal Server Error",
                    "An error occurred while serving this page."
                ),
                content_type='text/html',
                status=500
            )
    
    async def _fetch_from_dag(self, manifest_hash: str, path: str) -> tuple[bytes, str]:
        """
        Fetch file from DAG storage using manifest.
        
        Args:
            manifest_hash: Root manifest block hash
            path: File path within website
        
        Returns:
            Tuple of (content_bytes, content_type)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ConnectionError: If DAG retrieval fails
        """
        try:
            # Retrieve manifest block
            manifest_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.dag.retrieve(manifest_hash)
            )
            
            # Parse manifest JSON
            manifest = json.loads(manifest_data.decode('utf-8'))
            
            if manifest.get('type') != 'website_manifest':
                raise ValueError(f"Invalid manifest type: {manifest.get('type')}")
            
            # Look up file in manifest
            files = manifest.get('files', {})
            file_info = files.get(path)
            
            if not file_info:
                raise FileNotFoundError(f"File not found in manifest: {path}")
            
            # Retrieve file content from DAG
            file_hash = file_info['hash']
            content = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.dag.retrieve(file_hash)
            )
            
            content_type = file_info.get('content_type', 'application/octet-stream')
            
            return content, content_type
            
        except json.JSONDecodeError as e:
            raise ConnectionError(f"Invalid manifest format: {e}")
        except KeyError as e:
            raise FileNotFoundError(f"Malformed manifest: missing {e}")
        except Exception as e:
            logger.error(f"DAG retrieval error: {e}")
            raise ConnectionError(f"DAG retrieval failed: {e}")
    
    def _generate_error_page(self, domain: str, title: str, message: str) -> str:
        """Generate HTML error page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {domain}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 20px;
            padding: 60px 40px;
            max-width: 600px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 20px;
        }}
        p {{
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
            line-height: 1.6;
        }}
        .domain {{
            font-family: 'Courier New', monospace;
            background: #f0f0f0;
            padding: 10px 20px;
            border-radius: 8px;
            display: inline-block;
            margin: 20px 0;
            color: #764ba2;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            color: #999;
            font-size: 0.9em;
        }}
        .logo {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🌐</div>
        <h1>{title}</h1>
        <div class="domain">{domain}</div>
        <p>{message}</p>
        <div class="footer">
            Powered by BelizeChain BNS & Pakit DAG Storage<br>
            Sovereign decentralized hosting
        </div>
    </div>
</body>
</html>"""
    
    def clear_cache(self):
        """Clear request cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    async def health_check(self) -> Dict:
        """
        Get service health status.
        
        Returns:
            Dictionary with health metrics
        """
        return {
            'status': 'healthy',
            'cached_files': len(self.cache),
            'backend': 'dag',
            'hosted_domains': len(self.manager.list_hosted_domains())
        }


async def run_hosting_service(
    website_manager,
    dag_backend,
    host: str = "127.0.0.1",
    port: int = 8000
):
    """
    Run hosting service (convenience function).
    
    Args:
        website_manager: WebsiteManager instance
        dag_backend: DAG storage backend
        host: Listen address (use 0.0.0.0 for Docker/cloud, set via PAKIT_HOST env var)
        port: Listen port
    """
    service = HostingService(website_manager, dag_backend, host, port)
    await service.start()
    
    try:
        # Keep running until interrupted
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await service.stop()
