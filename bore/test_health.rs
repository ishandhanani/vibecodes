use std::net::TcpStream;
use std::time::Duration;

fn main() {
    let port = 8000;
    let addr = format!("127.0.0.1:{}", port);
    
    println!("Testing connection to {}", addr);
    
    match TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(100)) {
        Ok(stream) => {
            println!("✓ Connect succeeded");
            
            if let Err(e) = stream.set_read_timeout(Some(Duration::from_millis(50))) {
                println!("✗ Failed to set timeout: {}", e);
                return;
            }
            
            let mut buf = [0u8; 1];
            match stream.peek(&mut buf) {
                Ok(n) => println!("✓ Peek succeeded, read {} bytes", n),
                Err(e) => println!("✗ Peek failed: {} (kind: {:?})", e, e.kind()),
            }
            
            // Try shutdown to see if it errors
            match stream.shutdown(std::net::Shutdown::Both) {
                Ok(_) => println!("✓ Shutdown succeeded"),
                Err(e) => println!("✗ Shutdown failed: {}", e),
            }
        }
        Err(e) => {
            println!("✗ Connect failed: {} (kind: {:?})", e, e.kind());
        }
    }
}
