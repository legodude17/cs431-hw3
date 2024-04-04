import sys
from typing import Self, NewType
from enum import StrEnum
import math

RepId = NewType("RepId", str)
Party = StrEnum('Party', ['R', 'D'])

type Rep = tuple[RepId, Party, str]

def read_data(filename: str) -> list[Rep]:
  """
  Read a data file from a tsv
  """
  with open(filename) as f:
    return [tuple(line.strip().split("\t")) for line in f]
  
def analyze_data(data: list[Rep], valid_issues: list[int]):
  """
  Analyze the data and determine if all the items have a single voting record or party, plus the majority party
  """
  partyAll = data[0][1]
  votingAll = []
  # We only want to consider the valid issues
  for i in valid_issues:
    votingAll.append(data[0][2][i])
  rCount = 0
  dCount = 0
  for _,party,voting in data:
    if party != partyAll:
      partyAll = None
    if votingAll != None:
      for j in range(len(valid_issues)):
        if votingAll[j] != voting[valid_issues[j]]:
          votingAll = None
          break
    if party == "D":
      dCount += 1
    elif party == "R":
      rCount += 1
    else:
      raise ValueError("Invalid party: " + party)
  # majority is None if there's a tie
  majority: Party | None = None
  if rCount > dCount:
    majority = "R"
  elif dCount > rCount:
    majority = "D"

  return partyAll, votingAll, majority

def calculate_entropy(data: list[Rep]):
  """
  Calculate the entropy of a data set
  """
  if len(data) == 0:
    return 0
  
  rCount = 0
  dCount = 0
  for _,party,_ in data:
    if party == "D":
      dCount += 1
    elif party == "R":
      rCount += 1
    else:
      raise ValueError("Invalid party: " + party)
  
  rCount /= len(data)
  dCount /= len(data)

  # We define 0 * log(0) to be 0, but python doesn't know that, so need to check for it
  if rCount == 0:
    return -dCount * math.log2(dCount)
  elif dCount == 0:
    return -rCount * math.log2(rCount)
  else:
    return -dCount * math.log2(dCount) - rCount * math.log2(rCount)
  
def split_data(data: list[Rep], issue: int):
  """
  Split data based on how each rep voted on a given issue
  """
  yes: list[Rep] = []
  no: list[Rep] = []
  present: list[Rep] = []

  for rep in data:
    if rep[2][issue] == "+":
      yes.append(rep)
    elif rep[2][issue] == "-":
      no.append(rep)
    elif rep[2][issue] == ".":
      present.append(rep)
    else:
      raise ValueError("Invalid vote: " + rep[2][issue])
  
  return yes, no, present
  
class DecisionNode:
  """
  A node in the decision tree
  """
  # The majority at this node
  # If there's a tie, it's the parent's majority
  # This is used to prune the tree
  majority: Party;
  # Leaf
  value: Party = None; # What party this node classifies as
  # Non-Leaf
  issue: int; # The issue this node splits on
  childFor: Self;
  childPresent: Self;
  childAgainst: Self;

  def is_leaf(self):
    """
    Returns whether or not this node is a leaf
    """
    return self.value != None

  @staticmethod
  def create_node(data: list[Rep], parent_majority: Party = None, valid_issues: list[int] = None) -> Self:
    """
    Create a decision tree from data. Calls itself recursivly to fill out the tree.
    """
    node = DecisionNode()
    # If there's no data, just make a leaf with the parent's majority
    if len(data) == 0:
      node.value = parent_majority
      return node
    
    # Initialize valid issues
    if valid_issues == None:
      valid_issues = list(range(len(data[0][2])))
    
    # Analayze the data and assign the node's majority
    partyAll, votingAll, majority = analyze_data(data, valid_issues)
    if majority == None:
      majority = parent_majority

    node.majority = majority
    
    if partyAll != None:
      # All the same party, classify it as that
      node.value = partyAll
      return node
    if votingAll != None:
      # All the same voting record, classify as the majority
      node.value = majority
      return node
    
    # Calculate the issue with the best gain
    totalEntropy = calculate_entropy(data)
    bestGain = -1
    bestIssue = -1
    for i in valid_issues:
      gain = totalEntropy
      yes, no, present = split_data(data, i)
      gain -= (len(yes) / len(data)) * calculate_entropy(yes)
      gain -= (len(no) / len(data)) * calculate_entropy(no)
      gain -= (len(present) / len(data)) * calculate_entropy(present)
      if gain > bestGain:
        bestGain = gain
        bestIssue = i
        
    # If we found a best issue, create a node the splits on it
    if bestIssue > -1:
      yes, no, present = split_data(data, bestIssue)
      # We need to ensure this node's children don't consider
      # the issue that this node split one,
      # otherwise it will recurse forever
      child_issues = valid_issues.copy()
      child_issues.remove(bestIssue)
      node.issue = bestIssue
      node.childFor = DecisionNode.create_node(yes, majority, child_issues)
      node.childAgainst = DecisionNode.create_node(no, majority, child_issues)
      node.childPresent = DecisionNode.create_node(present, majority, child_issues)
      return node
    
    raise IndexError("No issue with non-negative gain found, this shouldn't be possible")

  def classify(self, item: str) -> Party:
    """
    Classify a voting record using this tree
    """
    if self.is_leaf():
      # If a leaf, just return the value
      return self.value
    else:
      # Based on the vote on the issue, defer to one of the children
      vote = item[self.issue]
      if vote == "+":
        return self.childFor.classify(item)
      elif vote == "-":
        return self.childAgainst.classify(item)
      elif vote == ".":
        return self.childPresent.classify(item)
      else:
        raise ValueError("Invalid vote: " + vote)

  def print(self, level = 2):
    """
    Print this tree
    """
    if self.is_leaf():
      print(self.value)
      return
    
    indent = " " * level
    # Since 65 = "A" and the issues start at 0, this will convert the number to a letter
    print("Issue " + chr(65 + self.issue) + ":")
    print(indent + "+", end=" ")
    self.childFor.print(level + 2)
    print(indent + "-", end=" ")
    self.childAgainst.print(level + 2)
    print(indent + ".", end=" ")
    self.childPresent.print(level + 2)

  def clone(self) -> Self:
    """
    Clone this tree deeply
    """
    node = DecisionNode()
    node.data = self.data.copy()
    if self.is_leaf():
      node.value = self.value
    else:
      node.issue = self.issue
      node.childFor = self.childFor.clone()
      node.childAgainst = self.childAgainst.clone()
      node.childPresent = self.childPresent.clone()

    return node

def calc_accuracy(tree: DecisionNode, data: list[Rep]):
  """
  Calculate the accuracy of a given tree on the data
  """
  numRight = 0
  for _,party,votes in data:
    if tree.classify(votes) == party:
      numRight += 1
  return numRight / len(data)

def count_children(node: DecisionNode):
  """
  Count the number of children of a tree
  """
  if node.is_leaf():
    return 1
  
  return count_children(node.childFor) + count_children(node.childAgainst) + count_children(node.childPresent) + 1

def best_prune(base: DecisionNode, bestAccuracy: float, allData: list[Rep], cur: DecisionNode = None):
  """
  Determine the best prune, which is the one that increases accuracy the most.
  Ties are broken by number of nodes eliminated
  """
  if cur == None:
    cur = base
  finalNode = None
  bestCount = 0
  # We can't prune a leaf, so just return
  if cur.is_leaf():
    return bestAccuracy, finalNode, bestCount

  # First check the for branch, find the best prune of that subtree
  forAcc, forNode, forCount = best_prune(base, bestAccuracy, allData, cur.childFor)
  # If it's better than what we currently have, select it
  if forNode != None and forAcc > bestAccuracy or (forAcc == bestAccuracy and forCount > bestCount):
    bestAccuracy = forAcc
    finalNode = forNode
    bestCount = forCount
  
  antiAcc, antiNode, antiCount = best_prune(base, bestAccuracy, allData, cur.childAgainst)
  if antiNode != None and antiAcc > bestAccuracy or (antiAcc == bestAccuracy and antiCount > bestCount):
    bestAccuracy = antiAcc
    finalNode = antiNode
    bestCount = antiCount

  presAcc, presNode, presCount = best_prune(base, bestAccuracy, allData, cur.childPresent)
  if presNode != None and presAcc > bestAccuracy or (forAcc == bestAccuracy and presCount > bestCount):
    bestAccuracy = presAcc
    finalNode = presNode
    bestCount = presCount

  # Finally, we check the results of pruning the current node
  # We can temporarily prune the node by setting the value then unsetting
  # Since the children are still set, those nodes won't be GC'd
  cur.value = cur.majority
  acc = calc_accuracy(base, allData)
  count = count_children(cur)
  cur.value = None
  # If it's better than our current, select it
  if acc > bestAccuracy or (acc == bestAccuracy and count > bestCount):
    bestAccuracy = acc
    finalNode = cur
    bestCount = count

  return bestAccuracy, finalNode, bestCount

def create_prune(data: list[Rep]) -> DecisionNode:
  """
  Create a decision tree from data, using every fourth element as a tuning set to prune the tree.
  """
  # Separate training and tuning data
  training = []
  tuning = []
  for i in range(len(data)):
    if i % 4 == 0:
      tuning.append(data[i])
    else:
      training.append(data[i])
  
  # Create the tree, get the accuracy on the tuning data
  tree = DecisionNode.create_node(training)
  bestAccuracy = calc_accuracy(tree, tuning)

  # Keep finding the best prune until they're all worse
  while True:
    bestAccuracy, to_prune,_ = best_prune(tree, bestAccuracy, tuning, tree)
    if to_prune == None:
      break
    # We can prune a node just by setting it's value,
    # since it will ignore the children if value is set
    to_prune.value = to_prune.majority

  return tree

# MAIN CODE
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python " + sys.argv[0] + " <file>")
    sys.exit(1)

  # Read data from the given file
  data = read_data(sys.argv[1])

  # Step 1: Create a single tree w/ pruning
  tree = create_prune(data)
  tree.print()

  # Step 2: Estimate accuracy
  rightCount = 0
  for item in data.copy():
    # Remove the item, create a tree with the remainder,
    # test the tree on the item, then add it back
    data.remove(item)
    tree = create_prune(data)
    if tree.classify(item[2]) == item[1]:
      rightCount += 1
    data.append(item)

  print(f"Accuracy: {rightCount / len(data)}")
