

#  rerank更关注的是文档的相关性，而不是文档的内容相似度。所以与payment_reference_code更相关的是invoice no，而不是reference number


import cohere
import yaml


class CohereRerankProcess:

    def rerank(self):
        co = cohere.ClientV2(api_key="")

        # Define the documents
        # documents = [
        #     "RV400005824",
        #     "4000058247",
        #     "5111156235",
        #     'RV240207021'
        # ]

        # documents = [
        #     "Invoice No",
        #     "Reference Number",
        #     "Reference",
        #     "Number",
        #     "PO NO",
        #     "GRN No"
        # ]

        # documents = [
        #     "Invoice No-RV240207021",
        #     "Invoice No-5111156235",
        #     "Reference Number-5111156235",
        #     "Reference Number-4000058247"
        # ]

        #
        # documents = [
        #     "Date-12.11.2024",
        #     "-12.11.2024",
        #     "Payment Date-12.11.2024",
        #     "Payment-12.11.2024",
        # ]

        # documents = [
        #     {
        #         "Invoice No": "RV240207021"
        #     },
        #     {
        #         "Invoice No": "5111156235"
        #     },
        #     {
        #         "Reference Number": "5111156235"
        #     }
        # ]

        # documents = [
        #     {
        #         "Invoice No": "RV240207021"
        #     },
        #     {
        #         "Invoice No": "5111156235"
        #     },
        #     {
        #         "Reference Number": "5111156235"
        #     }
        # ]

        # documents = [
        #     "None-None#Reference Number-RV240207021#Date-12.11.2024#Total-158,765,479.00",
        #     "None-None#Invoice No-RV240207021#Date-12.11.2024#Total-158,765,479.00",
        #     "None-None#Reference Number-5111156235#None-12.11.2024#Payment amt-158,765,479.00",
        # ]

        documents = [
            {
                "paymentCustomerName": {
                    "sourceField": None,
                    "value": None
                },
                "paymentNote": {
                    "sourceField": "Reference Number",
                    "value": "RV240207021"
                },
                "paymentDate": {
                    "sourceField": None,
                    "value": "12.11.2024"
                },
                "paymentAmount": {
                    "sourceField": "Total",
                    "value": "158,765,479.00"
                }
            },
            {
                "paymentCustomerName": {
                    "sourceField": None,
                    "value": None
                },
                "paymentNote": {
                    "sourceField": "Reference Number",
                    "value": "5111156235"
                },
                "paymentDate": {
                    "sourceField": None,
                    "value": "12.11.2024"
                },
                "paymentAmount": {
                    "sourceField": "Payment amt",
                    "value": "158,765,479.00"
                }
            }
        ]

        # Define the documents
        # documents = [
        #     {
        #         "from": "hr@co1t.com",
        #         "to": "david@co1t.com",
        #         "date": "2024-06-24",
        #         "subject": "A Warm Welcome to Co1t!",
        #         "text": "We are delighted to welcome you to the team! As you embark on your journey with us, you'll find attached an agenda to guide you through your first week.",
        #     },
        #     {
        #         "from": "it@co1t.com",
        #         "to": "david@co1t.com",
        #         "date": "2024-06-24",
        #         "subject": "Setting Up Your IT Needs",
        #         "text": "Greetings! To ensure a seamless start, please refer to the attached comprehensive guide, which will assist you in setting up all your work accounts.",
        #     },
        #     {
        #         "from": "john@co1t.com",
        #         "to": "david@co1t.com",
        #         "date": "2024-06-24",
        #         "subject": "First Week Check-In",
        #         "text": "Hello! I hope you're settling in well. Let's connect briefly tomorrow to discuss how your first week has been going. Also, make sure to join us for a welcoming lunch this Thursday at noon—it's a great opportunity to get to know your colleagues!",
        #     },
        # ]

        # Convert the documents to YAML format

        yaml_docs = [yaml.dump(doc, sort_keys=False) for doc in documents]

        print(yaml_docs, end="\n---------------------------\n")

        # Add the user query

        # query = "None-None#Reference Number-4000058247#None-12.11.2024#Payment amt-158,765,479.00"
        # query = "payment_reference_code:\n  sourceField: Reference Number\n  value: '4000058247'\payment_amount:\n  sourceField: Payment amt\n  value: 158,765,479.00\npayment_date:\n  sourceField: null\n  value: 12.11.2024\n"
        # query = "the reference code of the payment"
        # query = "paymentNote"
        query = "the billing reference of the related order"
        # query = "Reference Number: 4000058247"

        # query = "Reference Number-4000058247"
        # query = "Reference"
        # query = "4000058247"

        # query = "-12.11.2024"

        # Rerank the documents

        results = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=yaml_docs,
            top_n=4,
        )

        print(results)

        # Display the reranking results

        def return_results(results, documents):
            for idx, result in enumerate(results.results):
                print(f"Rank: {idx + 1}")
                print(f"Score: {result.relevance_score}")
                print(f"Document: {documents[result.index]}\n")

        return_results(results, documents)




if __name__ == '__main__':
    CohereRerankProcess().rerank()