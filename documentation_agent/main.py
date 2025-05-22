import operator
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# .envファイルから環境変数を読み込む
load_dotenv()

#ペルソナを表すデータモデル
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="パーソンの背景")

#ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas : list[Persona] = Field(
        default_factory=list, description="ペルソナのリスト"
    )

#インタビュー内容を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューの質問")
    answer: str = Field(..., description="インタビューの回答")

#インタビューの結果を表すデータモデル
class InterviewResult(BaseModel):
    interviews : list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )

#評価の結果を表すデータモデル
class EvaluationResult(BaseModel):
    reason : str = Field(..., description="判断の理由")
    is_sufficient : bool = Field(..., description="情報が十分かどうか")


class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューリスト"
    ) 
    requirements_doc: str =Field(default="", description="生成された要件定義")
    iteration: int = Field(
        default=0, description="ペルソナ生成とインタビューの反復回数"
    )
    is_inforamation_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )
    
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int=5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k
    def run(self, user_request: str) -> Personas:
        #プロンプトテンプレ
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーインタビュー用の多様なペルソナを作成する専門家です。",
                ),
                (
                    "human",
                    f"以下のユーザーリクエストに関するインタビューように{self.k}人の多様なペルソナを生成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "各ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。"
                ),
            ]
        )
        #ペルソナ作成チェーン
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})
    
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        answers = self._generate_answers(personas=personas, questions=questions)
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )
        return InterviewResult(interviews=interviews)
    def _generate_questions(
            self, user_request:str, personas: list[Persona]
        ) -> list[str]:
        #質問生成プロンプトの定義
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "あなたはユーザー要件に基づいて適切な質問を生成する専門家です",
                ),
                (
                    "human",
                    "以下のペルソナに関連するユーザーリクエストについて、一つ質問を生成してください\n\n"
                    "ユーザーリクエスト: {user_request}\n"
                    "ペルソナ: {persona_name} - {persona_background}\n\n"
                    "質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください",
                ),
            ]
        )
        #質問生成チェーン
        question_chain = question_prompt | self.llm | StrOutputParser()

        #各ペルソナに対する質問クエリを作成
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        #質問をバッチ処理
        return question_chain.batch(question_queries)

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
    #回答生成プロンプト定義
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナとして回答しています: {persona_name} - {persona_background}",
                ),
                (
                    "human","質問: {question}"
                ),
            ]
        )
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        return answer_chain.batch(answer_queries)
    
    def _create_interviews(self, personas: list[Persona], questions: list[str], answers: list[str]) -> list[Interview]:
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]

class InforamtionEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)
    
    #ユーザーリクエストとインタビューの評価
    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。",
                ),
                (
                    "human",
                    "以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成っするのに十分な情報が集まったかどうかを判断してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "インタビュー結果: \n{interview_results}",
                ),
            ]
        )
        #情報の十分性評価のチェーン作成
        chain = prompt | self.llm
        #評価結果を返す
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n"
                    f"回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )
        

class RequiementDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは収集した情報に基づいて要件文書を作成する専門家です。",
                ),
                (
                    "human",
                    "以下のユーザーリクエストと複数のペルソナからのインタビュー結果に基づいて要件文書を作成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "インタビュー結果: \n{interview_results}\n"
                    "要件文書には以下のセクションを含めてください:\n"
                    "1. プロジェクト概要\n"
                    "2. 主要機能\n"
                    "3. 非機能要件\n"
                    "4. 制約条件\n"
                    "5. ターゲットユーザー\n"
                    "6. 優先順位\n"
                    "7. リスクと軽減策\n"
                    "出力は必ず日本語でお願いします。\n\n要件文書:",
                ),
            ]
        )
        #要件定義書を生成するチェーンを作成
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n"
                    f"回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

class DocumentationAgent:
    def __init__(self, llm:ChatOpenAI, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InforamtionEvaluator(llm=llm)
        self.requirements_generator = RequiementDocumentGenerator(llm=llm)

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewState)

        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        workflow.set_entry_point("generate_personas")

        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.interation <5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)

        return workflow.compile()
    
    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }
    
    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }
    
    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_generator.run(
            state.user_request, state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state["requirements_doc"]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ユーザー要求に基づいて要件定義を作成します"
    )

    #task引数
    parser.add_argument(
        "--task",
        type=str,
        help="作成したいアプリケーションについて記載してください"
    )
    # K引数
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成するペルソナの人数を設定してください"
    )

    args = parser.parse_args()

    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.0)
    agent = DocumentationAgent(llm=llm, k=args.k)

    final_output = agent.run(user_request=args.task)
    print(final_output)

if __name__ == "__main__":
    main()