import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, List

import graphviz

directory = Path(".")


# Define AST node classes
@dataclass
class Number:
    value: int


@dataclass
class Boolean:
    value: bool


@dataclass
class Identifier:
    name: str


@dataclass
class StringLiteral:
    value: str


@dataclass
class Application:
    function: Any
    argument: Any


@dataclass
class Lambda:
    parameter: str
    body: Any


@dataclass
class Let:
    name: str
    value: Any
    body: Any


@dataclass
class If:
    condition: Any
    then_branch: Any
    else_branch: Any


@dataclass
class List:
    elements: List[Any]


@dataclass
class Operator:
    symbol: str


class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        if self.value:
            return f"Token({self.type}, {self.value})"
        return f"Token({self.type})"


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if len(self.text) > 0 else None

    def advance(self):
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        # Skip until end of line for single line comments
        if self.current_char == "-" and self.peek() == "-":
            while self.current_char is not None and self.current_char != "\n":
                self.advance()
            if self.current_char == "\n":
                self.advance()

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]

    def number(self):
        result = ""
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def identifier(self):
        result = ""
        if self.current_char.islower() or self.current_char == "_":
            result += self.current_char
            self.advance()
            while self.current_char is not None and (
                self.current_char.isalnum() or self.current_char == "_"
            ):
                result += self.current_char
                self.advance()
            return result
        else:
            raise Exception(f"Invalid identifier: {self.current_char}")

    def string(self):
        result = ""
        # Skip the opening quote
        self.advance()
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == "\\" and self.peek() == '"':
                self.advance()  # Skip the backslash
            result += self.current_char
            self.advance()
        # Skip the closing quote
        if self.current_char == '"':
            self.advance()
        else:
            raise Exception("Unterminated string")
        return result

    def operator(self):
        """Handle basic operators like +, -, *, /, >, <, ==, etc."""
        # List of supported operators
        operators = ["+", "-", "*", "/", ">", "<", "==", ">=", "<=", "/=", "&&", "||"]

        # First try multi-character operators
        for op in operators:
            if len(op) > 1 and self.text[self.pos : self.pos + len(op)] == op:
                result = op
                for _ in range(len(op)):
                    self.advance()
                return result

        # Then try single-character operators
        if self.current_char in "+-*/><&|":
            result = self.current_char
            self.advance()
            return result

        return None

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == "-" and self.peek() == "-":
                self.skip_comment()
                continue

            if self.current_char.isdigit():
                return Token("NUMBER", self.number())

            if self.current_char.islower() or self.current_char == "_":
                ident = self.identifier()
                if ident == "let":
                    return Token("LET")
                elif ident == "in":
                    return Token("IN")
                elif ident == "if":
                    return Token("IF")
                elif ident == "then":
                    return Token("THEN")
                elif ident == "else":
                    return Token("ELSE")
                elif ident == "True":
                    return Token("BOOLEAN", True)
                elif ident == "False":
                    return Token("BOOLEAN", False)
                else:
                    return Token("ID", ident)

            if self.current_char == "\\":
                self.advance()
                return Token("LAMBDA")

            if self.current_char == ".":
                self.advance()
                return Token("DOT")

            if self.current_char == "=":
                if self.peek() == "=":
                    self.advance()
                    self.advance()
                    return Token("OPERATOR", "==")
                self.advance()
                return Token("EQUALS")

            if self.current_char == "(":
                self.advance()
                return Token("LPAREN")

            if self.current_char == ")":
                self.advance()
                return Token("RPAREN")

            if self.current_char == "[":
                self.advance()
                return Token("LBRACKET")

            if self.current_char == "]":
                self.advance()
                return Token("RBRACKET")

            if self.current_char == ",":
                self.advance()
                return Token("COMMA")

            if self.current_char == '"':
                return Token("STRING", self.string())

            # Check for operators
            op = self.operator()
            if op:
                return Token("OPERATOR", op)

            raise Exception(f"Unexpected character: {self.current_char}")

        return Token("EOF")


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type):
        if self.current_token.type == token_type:
            result = self.current_token
            self.current_token = self.lexer.get_next_token()
            return result
        else:
            raise Exception(f"Expected {token_type}, got {self.current_token}")

    def parse(self):
        return self.expr()

    def expr(self):
        """
        expr : let_expr
             | if_expr
             | lambda_expr
             | application
        """
        if self.current_token.type == "LET":
            return self.let_expr()
        elif self.current_token.type == "IF":
            return self.if_expr()
        elif self.current_token.type == "LAMBDA":
            return self.lambda_expr()
        else:
            return self.application()

    def application(self):
        """
        application : atom application_tail
        """
        node = self.atom()

        # Continue parsing applications as long as we have valid atoms
        while self.current_token.type in (
            "NUMBER",
            "ID",
            "LPAREN",
            "STRING",
            "BOOLEAN",
            "LBRACKET",
        ):
            arg = self.atom()
            node = Application(node, arg)

        # Handle operators
        if self.current_token.type == "OPERATOR":
            op_token = self.eat("OPERATOR")
            op = Operator(op_token.value)
            right = self.expr()  # Parse the right side of the operator

            # In Haskell, operators are just infix function applications
            # So (x + y) is equivalent to ((+) x y) or (+ x y) in prefix notation
            # We'll represent it as Application(Application(Operator(+), x), y)
            return Application(Application(op, node), right)

        return node

    def atom(self):
        """
        atom : NUMBER
             | ID
             | STRING
             | BOOLEAN
             | list_expr
             | OPERATOR
             | LPAREN expr RPAREN
        """
        if self.current_token.type == "NUMBER":
            token = self.eat("NUMBER")
            return Number(token.value)
        elif self.current_token.type == "ID":
            token = self.eat("ID")
            return Identifier(token.value)
        elif self.current_token.type == "STRING":
            token = self.eat("STRING")
            return StringLiteral(token.value)
        elif self.current_token.type == "BOOLEAN":
            token = self.eat("BOOLEAN")
            return Boolean(token.value)
        elif self.current_token.type == "OPERATOR":
            token = self.eat("OPERATOR")
            return Operator(token.value)
        elif self.current_token.type == "LBRACKET":
            return self.list_expr()
        elif self.current_token.type == "LPAREN":
            self.eat("LPAREN")
            # Check if we have an operator in parentheses
            if self.current_token.type == "OPERATOR":
                op = Operator(self.eat("OPERATOR").value)
                self.eat("RPAREN")
                return op
            else:
                node = self.expr()
                self.eat("RPAREN")
                return node
        else:
            raise Exception(f"Unexpected token in atom: {self.current_token}")

    def list_expr(self):
        """
        list_expr : LBRACKET (expr (COMMA expr)*)? RBRACKET
        """
        self.eat("LBRACKET")
        elements = []

        if self.current_token.type != "RBRACKET":
            elements.append(self.expr())

            while self.current_token.type == "COMMA":
                self.eat("COMMA")
                elements.append(self.expr())

        self.eat("RBRACKET")
        return List(elements)

    def lambda_expr(self):
        """
        lambda_expr : LAMBDA ID DOT expr
        """
        self.eat("LAMBDA")
        param = self.eat("ID").value
        self.eat("DOT")
        body = self.expr()
        return Lambda(param, body)

    def let_expr(self):
        """
        let_expr : LET ID EQUALS expr IN expr
        """
        self.eat("LET")
        name = self.eat("ID").value
        self.eat("EQUALS")
        value = self.expr()
        self.eat("IN")
        body = self.expr()
        return Let(name, value, body)

    def if_expr(self):
        """
        if_expr : IF expr THEN expr ELSE expr
        """
        self.eat("IF")
        condition = self.expr()
        self.eat("THEN")
        then_branch = self.expr()
        self.eat("ELSE")
        else_branch = self.expr()
        return If(condition, then_branch, else_branch)


def visualize_ast(node, indent=0):
    """Pretty-print the AST for easier viewing."""
    prefix = "  " * indent

    if isinstance(node, Number):
        return f"{prefix}Number({node.value})"
    elif isinstance(node, Boolean):
        return f"{prefix}Boolean({node.value})"
    elif isinstance(node, Identifier):
        return f"{prefix}Identifier({node.name})"
    elif isinstance(node, StringLiteral):
        return f'{prefix}String("{node.value}")'
    elif isinstance(node, Operator):
        return f'{prefix}Operator("{node.symbol}")'
    elif isinstance(node, Application):
        return (
            f"{prefix}Application(\n"
            + f"{visualize_ast(node.function, indent + 1)},\n"
            + f"{visualize_ast(node.argument, indent + 1)}\n"
            + f"{prefix})"
        )
    elif isinstance(node, Lambda):
        return (
            f"{prefix}Lambda(\n"
            + f"{prefix}  parameter: {node.parameter},\n"
            + f"{prefix}  body: {visualize_ast(node.body, indent + 2)}\n"
            + f"{prefix})"
        )
    elif isinstance(node, Let):
        return (
            f"{prefix}Let(\n"
            + f"{prefix}  name: {node.name},\n"
            + f"{prefix}  value: {visualize_ast(node.value, indent + 2)},\n"
            + f"{prefix}  body: {visualize_ast(node.body, indent + 2)}\n"
            + f"{prefix})"
        )
    elif isinstance(node, If):
        return (
            f"{prefix}If(\n"
            + f"{prefix}  condition: {visualize_ast(node.condition, indent + 2)},\n"
            + f"{prefix}  then: {visualize_ast(node.then_branch, indent + 2)},\n"
            + f"{prefix}  else: {visualize_ast(node.else_branch, indent + 2)}\n"
            + f"{prefix})"
        )
    elif isinstance(node, List):
        elements = ",\n".join(
            [visualize_ast(elem, indent + 1) for elem in node.elements]
        )
        if elements:
            return f"{prefix}List([\n{elements}\n{prefix}])"
        else:
            return f"{prefix}List([])"
    else:
        return f"{prefix}Unknown({node})"


def parse_haskell(code):
    lexer = Lexer(code)
    parser = Parser(lexer)
    ast = parser.parse()
    return ast


def generate_ast_graph(node, filename="ast_graph"):
    dot = graphviz.Digraph(comment="AST Visualization")

    # Counter for unique node IDs
    counter = [0]

    def add_node(node, parent_id=None, edge_label=None):
        node_id = str(counter[0])
        counter[0] += 1

        if isinstance(node, Number):
            label = f"Number\n{node.value}"
        elif isinstance(node, Boolean):
            label = f"Boolean\n{node.value}"
        elif isinstance(node, Identifier):
            label = f"Identifier\n{node.name}"
        elif isinstance(node, StringLiteral):
            label = f'String\n"{node.value}"'
        elif isinstance(node, Operator):
            label = f'Operator\n"{node.symbol}"'
        elif isinstance(node, Application):
            label = "Application"
        elif isinstance(node, Lambda):
            label = f"Lambda\nparam: {node.parameter}"
        elif isinstance(node, Let):
            label = f"Let\nname: {node.name}"
        elif isinstance(node, If):
            label = "If"
        elif isinstance(node, List):
            label = "List"
        else:
            label = str(type(node).__name__)

        dot.node(node_id, label)

        if parent_id is not None:
            if edge_label:
                dot.edge(parent_id, node_id, label=edge_label)
            else:
                dot.edge(parent_id, node_id)

        if isinstance(node, Application):
            add_node(node.function, node_id, "function")
            add_node(node.argument, node_id, "argument")
        elif isinstance(node, Lambda):
            add_node(node.body, node_id, "body")
        elif isinstance(node, Let):
            add_node(node.value, node_id, "value")
            add_node(node.body, node_id, "body")
        elif isinstance(node, If):
            # Add clear labels for each part of the If expression
            add_node(node.condition, node_id, "condition")
            add_node(node.then_branch, node_id, "then")
            add_node(node.else_branch, node_id, "else")
        elif isinstance(node, List):
            for i, elem in enumerate(node.elements):
                add_node(elem, node_id, f"element[{i}]")

        return node_id

    add_node(node)
    dot.render(filename, view=False, format="png")
    for file in directory.iterdir():
        if file.is_file() and file.suffix == "":
            print(f"Deleting: {file.name}")
            file.unlink()

    return f"Parse tree saved as {filename}.png"


def create_gui():
    window = tk.Tk()
    window.title("Haskell Expression Parser")
    window.geometry("800x600")

    # Input frame
    input_frame = ttk.LabelFrame(window, text="Enter Haskell Expression")
    input_frame.pack(fill="both", expand=True, padx=10, pady=10)

    text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=5)
    text_input.pack(fill="both", expand=True, padx=5, pady=5)

    # Output frame
    output_frame = ttk.LabelFrame(window, text="AST Visualization")
    output_frame.pack(fill="both", expand=True, padx=10, pady=10)

    output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
    output_text.pack(fill="both", expand=True, padx=5, pady=5)

    # Parse button
    def parse_expression():
        try:
            code = text_input.get(1.0, tk.END).strip()
            if not code:
                messagebox.showwarning("Warning", "Please enter an expression first")
                return

            ast = parse_haskell(code)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, visualize_ast(ast))

            # Generate and show the AST graph
            filename = "gui_ast_output"
            generate_ast_graph(ast, filename)
            messagebox.showinfo("Success", f"AST graph generated as {filename}.png")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse: {str(e)}")

    parse_btn = ttk.Button(window, text="Parse Expression", command=parse_expression)
    parse_btn.pack(pady=10)

    window.mainloop()


# Example usage
if __name__ == "__main__":
    # # Example 1: Simple function application
    # code1 = "add 5 10"

    # # Example 2: Lambda expression
    # code2 = "\\x. x + 1"

    # # Example 3: Let expression
    # code3 = "let x = 42 in x * 2"

    # # Example 4: If expression
    # code4 = 'if x > 10 then "large" else "small"'

    # # Example 5: List
    # code5 = "[1, 2, 3]"

    create_gui()
