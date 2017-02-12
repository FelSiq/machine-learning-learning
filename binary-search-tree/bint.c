#include <stdlib.h>
#include <stdio.h>
#include <time.h> //Para testes aleatorios
#include "bint.h"

typedef struct node {
	int key;
	struct node *sonLeft, *sonRight;
	ITEM *item;
} NODE;

struct binT{
	NODE *root;
};

//Cria a estrutura inicial da arvore
binT *btInit(){
	binT *bt = malloc(sizeof(binT));
	if (bt != NULL){
		bt->root = NULL;
	}
	return bt;
};

//F. auxiliar da btCount
static unsigned int btCount_rec(NODE *root){
	if (root != NULL)
		return btCount_rec(root->sonLeft) + btCount_rec(root->sonRight) + 1;
	return 0;
};

//Conta quantos nós a árvore possui
unsigned int btCount(binT *bt){
	if (bt != NULL)
		return btCount_rec(bt->root);
	return 0;
};

//Função recursiva auxiliar da f. de inserção
//Note que não há tratamento explícito para chaves repetidas
static void btInsert_rec(NODE *root, NODE *node){
	//Verifica se a posição ideal está para a esquerda, i.e
	//A chave do novo elemento é MENOR que a chave da root atual
	if (root->key > node->key){
		//Verifica se encontramos a posição procurada, isto é,
		//se o filho da esquerda de root é inexistente
		if (root->sonLeft != NULL){
			btInsert_rec(root->sonLeft, node);
		} else {
			//Encontramos a posição ideal
			root->sonLeft = node;
		}
	} else {
		//Neste caso, temos que procurar para a DIREITA,
		//mesmo esquema.
		if (root->sonRight != NULL){
			btInsert_rec(root->sonRight, node);
		} else {
			root->sonRight = node;
		}
	}
}

//Inclue um item na arvore
void btInsert(binT *bt, unsigned int key, ITEM *item){
	if (bt != NULL){
		//Inicia uma novo nó, que contem "item", para ser inserido na árvore
		NODE *newNode = malloc(sizeof(NODE));
		newNode->sonLeft = NULL;
		newNode->sonRight = NULL;
		newNode->item = item;
		//A chave será usada para encontrar a posição do novo nó
		newNode->key = key;
		if (bt->root != NULL){
			//Neste caso, existe pelo menos um elemento inserido na árvore
			btInsert_rec(bt->root, newNode);
		} else {
			//Neste caso, estamos inserindo o primeiro elemento da árvore.
			bt->root = newNode;
		}
	}
};

//F. auxiliar da função de print
static void btPrint_rec(NODE *root, short int spaceNum){
	if (root != NULL){
		//Seção de print
		short int k = spaceNum;
		while(0 < k--)
			printf(" ");
		printf("%d\n", root->key);
		//Seção "descer para a esquerda"
		if (root->sonLeft != NULL){
			printf("l: ");
			btPrint_rec(root->sonLeft, spaceNum + 1);
		}
		//Seção "descer para a direita"
		if (root->sonRight != NULL){
			printf("r: ");
			btPrint_rec(root->sonRight, spaceNum + 1);
		}
	}
};

//Printa a arvore PRE-ORDER no modelo lista hierárquica
void btPrint(binT *bt){
	if (bt != NULL)
		btPrint_rec(bt->root, 0);
};

//F. auxiliar da função de busca
static ITEM *btSearch_rec(NODE *root, unsigned int key){
	//Verificar se não passamos além de uma folha da árvore
	if (root != NULL){
		if (key == root->key){
			//Encontramos o nó procurado, retorne seu ITEM
			return root->item;
		} else if (key > root->key){
			//Ainda ñ encontramos o nó, mas o mesmo está à DIREITA da root atual
			return btSearch_rec(root->sonRight, key);
		} else {
			//Só resta o nó estar à ESQUERDA da root atual.
			return btSearch_rec(root->sonLeft, key);
		}
	}

	//Caso chegue aqui, é por quê não encontramos o nó. 
	//retorne NULL.
	return NULL;
};

//Busca um nó da arvore com a key "key", e retorna seu "ITEM", se encontrado.
//Caso contrário, retorna NULL.
ITEM *btSearch(binT *bt, unsigned int key){
	if (bt != NULL)
		return btSearch_rec(bt->root, key);
	return NULL;
};

//Funções auxiliares da função de remoção
static NODE *btRemoveSubs(NODE *root){
	//Primeiro tentar obter o "maior dos menores", e, caso não seja possível,
	//tentar o "menor dos maiores".
	if (root != NULL){
		NODE *newNode = NULL, *traveller = NULL, *prevNode = NULL;

		//Se uma das condições abaixo for satisfeita, retornar o próprio root ao final
		if (root->sonLeft != NULL) {
			newNode = root;

			prevNode = NULL;
			traveller = root->sonLeft;

			while(traveller->sonRight != NULL){
				prevNode = traveller;
				traveller = traveller->sonRight;
			}

			if (prevNode != NULL)
				prevNode->sonRight = btRemoveSubs(traveller->sonLeft);
			else 
				root->sonLeft = traveller->sonLeft;

		} else if (root->sonRight != NULL) {
			newNode = root;

			prevNode = NULL;
			traveller = root->sonRight;

			while(traveller->sonLeft != NULL){
				prevNode = traveller;
				traveller = traveller->sonLeft;
			}

			if (prevNode != NULL)
				prevNode->sonLeft = btRemoveSubs(traveller->sonRight);
			else 
				root->sonRight = traveller->sonRight;
		}

		if (newNode == NULL){
			//Neste caso, o nó é uma raíz. Desalocar sua memória e retornar NULL.
			free(root);
		} else {
			//Se não, encontramos um candidato. Atualizar os dados de root e
			//desalocar a memória do antigo nó.
			root->item = traveller->item;
			root->key = traveller->key;
			//free(traveller);
		}
		
		return newNode;
	}
	return NULL;
};

static ITEM *btRemove_rec(NODE *root, unsigned int key, char *FLAG){
	ITEM *myItem = NULL;
	//Verificar se não fomos além de uma folha da árvore
	if (root != NULL){
		//Primeiro, procurar pelo nó desejado
		if (root->key == key){
			//Encontramos o nó, ligar flag e então retornar (ao fim da função)
			*FLAG = 1;
			myItem = root->item;
		} else {
			//Continuar procurando o nó desejado
			if (root->key > key){
				myItem = btRemove_rec(root->sonLeft, key, FLAG);
				if (*FLAG)
					root->sonLeft = btRemoveSubs(root->sonLeft);
			} else {
				myItem = btRemove_rec(root->sonRight, key, FLAG);
				if (*FLAG)
					root->sonRight = btRemoveSubs(root->sonRight);
			}
			//Desliga a FLAG
			*FLAG = 0;
			//Retorna o item recuperado
		}
	}

	return myItem;
};

//Remove um nó na arvore, e retorna seu item
//A removação é uma função um pouco complicada
//No geral, para se retirar um nó da árvore, que não seja uma folha,
//Deve-se substituí-lo, respeitando a hierarquia das keys impostas por uma
//ABB. 
//Assim, a solução é tentar trocar o nó pelo "maior dos menores", i.e, 
//Pegar o maior nó da sub-árvore esquerda do nó que será removido. Caso não
//exista uma sub-árvore á esquerda, então substituir pelo "menor dos maiores", i.e,
//o menor nó da sub-árvore á direita.
ITEM *btRemove(binT *bt, unsigned int key){
	ITEM *myItem = NULL;
	if (bt != NULL){
		//Esta flag avisará á função recursive que o nó desejado
		//foi encontrado, portanto, volte um nível de recursão e
		//procure um substituto válido.
		char FLAG = 0;
		myItem = btRemove_rec(bt->root, key, &FLAG);
		if (FLAG)
			bt->root = btRemoveSubs(bt->root);
	}
	return myItem;
};

//Função auxiliar da função de desalocação de memória dinâmica da árvore
static void btPurge_rec(NODE *root){
	if (root != NULL){
		//Tanto faz para qual lado você vai primeiro
		//durante a desalocação de mem. da árvore
		btPurge_rec(root->sonLeft);
		btPurge_rec(root->sonRight);
		free(root);
	}
};

//Destroi a arvore
void btPurge(binT *bt){
	if (bt != NULL){
		btPurge_rec(bt->root);
		free(bt);
	};
};

int main(int argc, char const *argv[]){
	//TESTES DAS FUNÇÕES IMPLEMENTADAS
	binT *bt = btInit();
	if (bt != NULL){
		//Gerar uma seed "aleatória"
		srand(time(NULL));
		int k;
		int myVector[15];

		for (k = 0; k < 15; k++){
			myVector[k] = rand() % 1000;
			btInsert(bt, myVector[k], NULL);
		}
		btPrint(bt);

		int t;
		for(k = 15; k > 0; k--){
			btPrint(bt);
			do t = (rand() % 15);
			while (myVector[t] < 0);
			printf("Key to remove: %d\tnodes on this tree: %u\n", myVector[t], btCount(bt));
			btRemove(bt, myVector[t]);
			myVector[t] = -1;
		}

		btPrint(bt);
		printf("nodes remaining: %u\n", btCount(bt));
		btPurge(bt);
	}	
	return 0;
}