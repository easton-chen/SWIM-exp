#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<arpa/inet.h>
#include<netinet/in.h>
#include<unistd.h>

int main() {
	int socket_desc;
	struct sockaddr_in server;
	char *message, server_reply[2000];
	
	// 创建socket
	socket_desc = socket(AF_INET, SOCK_STREAM, 0);
	if (-1 == socket_desc) {
		perror("connot create socket");
		exit(1);
	}
	
	server.sin_addr.s_addr = inet_addr("127.0.0.1");
	server.sin_family = AF_INET;
	server.sin_port = htons(50007);
	
	// 进行连接
	if (connect(socket_desc, (struct sockaddr*)&server, sizeof(server)) < 0) {
		perror("connot connect");
		return 1;
	}
	while(1){
        // 发送数据
        message = "hello world\r\n\r\n";
        if (send(socket_desc, message, strlen(message), 0) < 0) {
            perror("send data error");
            return 2;
        }
        
        printf("send message success\n");
        // 接收数据
        if (recv(socket_desc, server_reply, 2000, 0) < 0) {
            perror("recv error");
            return 3;
        }
        
        printf("recv success: ");
        puts(server_reply);
        sleep(1);
    }
	
	
	// 关闭socket
	close(socket_desc);
	return 0;
}
