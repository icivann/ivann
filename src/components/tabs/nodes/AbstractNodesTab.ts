import { SearchItem } from '@/components/SearchUtils';

export default abstract class AbstractNodesTab {
  abstract readonly searchItems: SearchItem[];
}
