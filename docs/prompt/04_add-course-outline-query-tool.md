In @backend/search_tools.py, add a second tool alongside the existing content-related tool. This new tool should handle course outline queries.
- Functionality:
  - Input: Course title
  - Output: Course title, course link, and complete lesson list
  - For each lesson: lesson number, lesson title
- Data source: Course metadata collection of the vector store
- Update the system prompt in @backend/ai_generator so that the course title, course link, the number and title of each lesson are all returned to address an outline-related queries.
- Make sure that the new tool is registered in the system.